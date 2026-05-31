# ============================================================
# main_individual_datacenter.py
# 데이터센터 개별 학습 + 시간별/일별/월별 예측
# Models: NBEATS, DeepAR, TFT
# Validation: 최근 6개월 holdout
# Forecasting: 24시간 rolling forecast
# ============================================================

# 필요 시 설치:
# pip install pandas numpy matplotlib scikit-learn torch neuralforecast lightning

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, DeepAR, TFT
from neuralforecast.losses.pytorch import MAE, DistributionLoss

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1) Reproducibility
# ------------------------------------------------------------
SEED = 42


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ------------------------------------------------------------
# 2) Config
# ------------------------------------------------------------
CSV_PATH = r"./data.csv"

OUT_DIR = Path("./forecast_outputs_individual")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FREQ = "H"
VAL_START = pd.Timestamp("2023-07-01 00:00:00")

# 24시간씩 rolling forecast
PRED_HORIZON = 24

# 입력 길이: 최근 1주
INPUT_SIZE = 24 * 7

# 학습 설정
MAX_STEPS = 100
BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
WINDOWS_BATCH_SIZE = 128
INFERENCE_WINDOWS_BATCH_SIZE = 128

USE_GPU = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_GPU else "cpu"
DEVICES = 1

# DeepAR 안정성 위해 FP32
PRECISION = "32-true"

PLOT_CONTEXT_HOURS = 24 * 14

# 너무 오래 걸리면 일부 센터만 테스트할 때 사용
LIMIT_CENTERS = None
# 예: LIMIT_CENTERS = 20


# ------------------------------------------------------------
# 3) Data loading / preprocessing
# ------------------------------------------------------------
def load_datacenter_data(csv_path: str) -> pd.DataFrame:
    """
    데이터센터 단위 시계열 생성
    unique_id = region_no
    """
    df = pd.read_csv(csv_path)

    required_cols = {"no", "region", "year", "month", "day", "hour", "elec_kwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df["ds"] = (
        pd.to_datetime(df[["year", "month", "day"]])
        + pd.to_timedelta(df["hour"] - 1, unit="h")
    )

    # 센터별 고유 ID
    df["unique_id"] = df["region"].astype(str) + "_" + df["no"].astype(str)

    ts = (
        df.groupby(["unique_id", "region", "no", "ds"], as_index=False)["elec_kwh"]
        .sum()
        .rename(columns={"elec_kwh": "y"})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )

    return ts


def repair_missing_hours_by_center(panel_df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    """
    센터별로 전체 시간축에 맞춰 reindex 후 보간
    """
    repaired = []

    global_start = panel_df["ds"].min()
    global_end = panel_df["ds"].max()
    full_index = pd.date_range(global_start, global_end, freq=freq)

    meta = (
        panel_df[["unique_id", "region", "no"]]
        .drop_duplicates()
        .set_index("unique_id")
    )

    for uid, g in panel_df.groupby("unique_id"):
        g = g.sort_values("ds").set_index("ds")[["y"]]
        g = g.reindex(full_index)

        missing_before = g["y"].isna().sum()
        if missing_before > 0:
            print(f"[INFO] {uid}: missing {missing_before} hours -> interpolated")

        g["y"] = g["y"].interpolate(method="time").bfill().ffill()
        g["unique_id"] = uid
        g["region"] = meta.loc[uid, "region"]
        g["no"] = meta.loc[uid, "no"]

        g = g.reset_index().rename(columns={"index": "ds"})
        repaired.append(g[["unique_id", "region", "no", "ds", "y"]])

    out = (
        pd.concat(repaired, axis=0)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return out


def add_time_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()

    df["hour"] = df["ds"].dt.hour
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["day"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


def split_train_valid(df: pd.DataFrame, val_start: pd.Timestamp):
    train_df = df[df["ds"] < val_start].copy()
    valid_df = df[df["ds"] >= val_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("train 또는 validation 데이터가 비어 있습니다.")

    return train_df, valid_df


# ------------------------------------------------------------
# 4) Metrics
# ------------------------------------------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(100.0 * np.abs((y_true - y_pred) / denom))


def smape(y_true, y_pred, eps=1e-8):
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(200.0 * np.abs(y_true - y_pred) / denom)


def calc_metrics(actual: pd.Series, pred: pd.Series):
    y_true = actual.values
    y_pred = pred.values
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "sMAPE(%)": smape(y_true, y_pred),
    }


# ------------------------------------------------------------
# 5) Model builder
# ------------------------------------------------------------
def build_models(futr_exog_list):
    trainer_kwargs = {
        "accelerator": ACCELERATOR,
        "devices": DEVICES,
        "precision": PRECISION,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "deterministic": False,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "log_every_n_steps": 1,
    }

    common_kwargs = dict(
        h=PRED_HORIZON,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        valid_batch_size=VALID_BATCH_SIZE,
        windows_batch_size=WINDOWS_BATCH_SIZE,
        inference_windows_batch_size=INFERENCE_WINDOWS_BATCH_SIZE,
        scaler_type="standard",
        random_seed=SEED,
        learning_rate=1e-4,
        **trainer_kwargs,
    )

    models = [
        NBEATS(
            **common_kwargs,
            stack_types=["trend", "seasonality"],
            n_blocks=[2, 2],
            mlp_units=[[256, 256], [256, 256]],
            loss=MAE(),
            valid_loss=MAE(),
        ),
        DeepAR(
            **common_kwargs,
            lstm_hidden_size=64,
            lstm_n_layers=2,
            lstm_dropout=0.1,
            decoder_hidden_layers=1,
            decoder_hidden_size=64,
            trajectory_samples=50,
            futr_exog_list=futr_exog_list,
            loss=DistributionLoss(distribution="Normal", level=[80, 90], return_params=False),
            valid_loss=MAE(),
        ),
        TFT(
            **common_kwargs,
            hidden_size=32,
            n_head=4,
            dropout=0.1,
            futr_exog_list=futr_exog_list,
            loss=MAE(),
            valid_loss=MAE(),
        ),
    ]
    return models


# ------------------------------------------------------------
# 6) Rolling forecast for one center
# ------------------------------------------------------------
def rolling_forecast_single_center(train_nf, valid_nf, futr_exog_list):
    models = build_models(futr_exog_list=futr_exog_list)
    model_names = [type(m).__name__ for m in models]

    nf = NeuralForecast(models=models, freq=FREQ)
    nf.fit(df=train_nf)

    history_df = train_nf.copy()
    pred_parts = []

    valid_start = valid_nf["ds"].min()
    valid_end = valid_nf["ds"].max()
    cutoff_points = pd.date_range(valid_start, valid_end, freq=f"{PRED_HORIZON}H")

    for cutoff in cutoff_points:
        future_slice = valid_nf[
            (valid_nf["ds"] >= cutoff) &
            (valid_nf["ds"] < cutoff + pd.Timedelta(hours=PRED_HORIZON))
        ][["unique_id", "ds"] + futr_exog_list].copy()

        if future_slice.empty:
            continue

        preds = nf.predict(df=history_df, futr_df=future_slice).reset_index()
        pred_parts.append(preds)

        actual_slice = valid_nf[
            (valid_nf["ds"] >= cutoff) &
            (valid_nf["ds"] < cutoff + pd.Timedelta(hours=PRED_HORIZON))
        ][["unique_id", "ds", "y"] + futr_exog_list].copy()

        history_df = pd.concat([history_df, actual_slice], axis=0, ignore_index=True)

    pred_df = pd.concat(pred_parts, axis=0, ignore_index=True)
    pred_df = pred_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return pred_df, model_names


# ------------------------------------------------------------
# 7) Aggregation
# ------------------------------------------------------------
def make_daily_monthly_predictions(hourly_valid_df: pd.DataFrame, hourly_pred_df: pd.DataFrame, model_names):
    merged = hourly_valid_df.merge(hourly_pred_df, on=["unique_id", "ds"], how="inner")

    # daily
    daily = merged.copy()
    daily["date"] = daily["ds"].dt.floor("D")

    daily_agg = (
        daily.groupby(["unique_id", "date"], as_index=False)
        .agg({"y": "sum", **{m: "sum" for m in model_names}})
        .rename(columns={"date": "ds"})
    )

    # monthly
    monthly = merged.copy()
    monthly["month_ds"] = monthly["ds"].dt.to_period("M").dt.to_timestamp()

    monthly_agg = (
        monthly.groupby(["unique_id", "month_ds"], as_index=False)
        .agg({"y": "sum", **{m: "sum" for m in model_names}})
        .rename(columns={"month_ds": "ds"})
    )

    return daily_agg, monthly_agg


# ------------------------------------------------------------
# 8) Plotting
# ------------------------------------------------------------
def save_hourly_plot(train_df, valid_pred_df, model_names, center_id, out_dir: Path):
    context = train_df.tail(PLOT_CONTEXT_HOURS)

    plt.figure(figsize=(16, 6))
    plt.plot(context["ds"], context["y"], label="train_recent_actual")
    plt.plot(valid_pred_df["ds"], valid_pred_df["y"], label="valid_actual")

    for m in model_names:
        plt.plot(valid_pred_df["ds"], valid_pred_df[m], label=m)

    plt.title(f"{center_id} | Hourly Validation Forecast")
    plt.xlabel("datetime")
    plt.ylabel("elec_kwh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{center_id}_hourly.png", dpi=150)
    plt.close()


def save_agg_plot(agg_df, model_names, center_id, level: str, out_dir: Path):
    plt.figure(figsize=(14, 5))
    plt.plot(agg_df["ds"], agg_df["y"], label=f"{level}_actual")
    for m in model_names:
        plt.plot(agg_df["ds"], agg_df[m], label=m)

    plt.title(f"{center_id} | {level.capitalize()} Forecast")
    plt.xlabel("datetime")
    plt.ylabel("elec_kwh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{center_id}_{level}.png", dpi=150)
    plt.close()


# ------------------------------------------------------------
# 9) Main
# ------------------------------------------------------------
def main():
    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if USE_GPU:
        print(f"[INFO] GPU name = {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] GPU not found. Running on CPU.")

    print("[INFO] Loading data...")
    panel = load_datacenter_data(CSV_PATH)
    print(f"[INFO] #raw centers = {panel['unique_id'].nunique()}")

    print("[INFO] Repairing missing hours by center...")
    panel = repair_missing_hours_by_center(panel, freq=FREQ)
    print(f"[INFO] #centers after repair = {panel['unique_id'].nunique()}")

    print("[INFO] Creating time features...")
    panel = add_time_features(panel)

    center_meta = (
        panel[["unique_id", "region", "no"]]
        .drop_duplicates()
        .sort_values(["region", "no"])
        .reset_index(drop=True)
    )

    centers = center_meta["unique_id"].tolist()
    if LIMIT_CENTERS is not None:
        centers = centers[:LIMIT_CENTERS]

    futr_exog_list = [
        "hour",
        "dayofweek",
        "day",
        "month",
        "weekofyear",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    all_hourly_preds = []
    all_daily_preds = []
    all_monthly_preds = []

    hourly_metrics_rows = []
    daily_metrics_rows = []
    monthly_metrics_rows = []

    for idx, center_id in enumerate(centers, 1):
        print(f"\n[INFO] ({idx}/{len(centers)}) Training center: {center_id}")

        center_df = panel[panel["unique_id"] == center_id].copy().sort_values("ds")
        train_df, valid_df = split_train_valid(center_df, VAL_START)

        train_nf = train_df[["unique_id", "ds", "y"] + futr_exog_list].copy()
        valid_nf = valid_df[["unique_id", "ds", "y"] + futr_exog_list].copy()

        try:
            pred_df, model_names = rolling_forecast_single_center(
                train_nf=train_nf,
                valid_nf=valid_nf,
                futr_exog_list=futr_exog_list,
            )
        except Exception as e:
            print(f"[WARN] {center_id} failed: {e}")
            continue

        merged_hourly = valid_nf[["unique_id", "ds", "y"]].merge(
            pred_df, on=["unique_id", "ds"], how="inner"
        )

        meta_row = center_meta[center_meta["unique_id"] == center_id].iloc[0]
        merged_hourly["region"] = meta_row["region"]
        merged_hourly["no"] = meta_row["no"]

        all_hourly_preds.append(merged_hourly)

        # hourly metrics
        for m in model_names:
            metric = calc_metrics(merged_hourly["y"], merged_hourly[m])
            hourly_metrics_rows.append({
                "unique_id": center_id,
                "region": meta_row["region"],
                "no": meta_row["no"],
                "model": m,
                **metric,
            })

        # daily / monthly aggregation
        daily_agg, monthly_agg = make_daily_monthly_predictions(
            hourly_valid_df=valid_nf[["unique_id", "ds", "y"]],
            hourly_pred_df=pred_df,
            model_names=model_names,
        )

        daily_agg["region"] = meta_row["region"]
        daily_agg["no"] = meta_row["no"]
        monthly_agg["region"] = meta_row["region"]
        monthly_agg["no"] = meta_row["no"]

        all_daily_preds.append(daily_agg)
        all_monthly_preds.append(monthly_agg)

        # daily metrics
        for m in model_names:
            metric = calc_metrics(daily_agg["y"], daily_agg[m])
            daily_metrics_rows.append({
                "unique_id": center_id,
                "region": meta_row["region"],
                "no": meta_row["no"],
                "model": m,
                **metric,
            })

        # monthly metrics
        for m in model_names:
            metric = calc_metrics(monthly_agg["y"], monthly_agg[m])
            monthly_metrics_rows.append({
                "unique_id": center_id,
                "region": meta_row["region"],
                "no": meta_row["no"],
                "model": m,
                **metric,
            })

        # plots
        save_hourly_plot(
            train_df=train_df[["ds", "y"]],
            valid_pred_df=merged_hourly[["ds", "y"] + model_names],
            model_names=model_names,
            center_id=center_id,
            out_dir=PLOT_DIR,
        )
        save_agg_plot(
            agg_df=daily_agg[["ds", "y"] + model_names],
            model_names=model_names,
            center_id=center_id,
            level="daily",
            out_dir=PLOT_DIR,
        )
        save_agg_plot(
            agg_df=monthly_agg[["ds", "y"] + model_names],
            model_names=model_names,
            center_id=center_id,
            level="monthly",
            out_dir=PLOT_DIR,
        )

    if not all_hourly_preds:
        raise RuntimeError("성공한 센터가 없습니다. 설정을 더 낮춰서 다시 시도하세요.")

    hourly_pred_all = pd.concat(all_hourly_preds, axis=0).reset_index(drop=True)
    daily_pred_all = pd.concat(all_daily_preds, axis=0).reset_index(drop=True)
    monthly_pred_all = pd.concat(all_monthly_preds, axis=0).reset_index(drop=True)

    hourly_metrics_df = pd.DataFrame(hourly_metrics_rows)
    daily_metrics_df = pd.DataFrame(daily_metrics_rows)
    monthly_metrics_df = pd.DataFrame(monthly_metrics_rows)

    # overall summary
    hourly_summary = hourly_metrics_df.groupby("model", as_index=False)[["MAE", "RMSE", "MAPE(%)", "sMAPE(%)"]].mean()
    daily_summary = daily_metrics_df.groupby("model", as_index=False)[["MAE", "RMSE", "MAPE(%)", "sMAPE(%)"]].mean()
    monthly_summary = monthly_metrics_df.groupby("model", as_index=False)[["MAE", "RMSE", "MAPE(%)", "sMAPE(%)"]].mean()

    # save
    hourly_pred_all.to_csv(OUT_DIR / "hourly_predictions_all_centers.csv", index=False, encoding="utf-8-sig")
    daily_pred_all.to_csv(OUT_DIR / "daily_predictions_all_centers.csv", index=False, encoding="utf-8-sig")
    monthly_pred_all.to_csv(OUT_DIR / "monthly_predictions_all_centers.csv", index=False, encoding="utf-8-sig")

    hourly_metrics_df.to_csv(OUT_DIR / "hourly_metrics_by_center.csv", index=False, encoding="utf-8-sig")
    daily_metrics_df.to_csv(OUT_DIR / "daily_metrics_by_center.csv", index=False, encoding="utf-8-sig")
    monthly_metrics_df.to_csv(OUT_DIR / "monthly_metrics_by_center.csv", index=False, encoding="utf-8-sig")

    hourly_summary.to_csv(OUT_DIR / "hourly_metrics_summary.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(OUT_DIR / "daily_metrics_summary.csv", index=False, encoding="utf-8-sig")
    monthly_summary.to_csv(OUT_DIR / "monthly_metrics_summary.csv", index=False, encoding="utf-8-sig")

    print("\n===== Hourly Summary =====")
    print(hourly_summary.sort_values("RMSE"))
    print("\n===== Daily Summary =====")
    print(daily_summary.sort_values("RMSE"))
    print("\n===== Monthly Summary =====")
    print(monthly_summary.sort_values("RMSE"))

    print("\n[INFO] Done.")
    print(f"[INFO] Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()