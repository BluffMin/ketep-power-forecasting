# ============================================================
# main.py
# Region-level hourly power forecasting with GPU
# Models: NBEATS, DeepAR, TFT
# Validation: last 6 months holdout
# Rolling forecast evaluation to avoid GPU OOM
# Stable DeepAR configuration to avoid NaN
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

OUT_DIR = Path("./forecast_outputs_gpu")
PLOT_DIR = OUT_DIR / "region_plots"
PER_MODEL_PLOT_DIR = PLOT_DIR / "per_model"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PER_MODEL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

FREQ = "H"
VAL_START = pd.Timestamp("2023-07-01 00:00:00")

# 6개월 전체를 한 번에 예측하지 않고 24시간씩 rolling forecast
PRED_HORIZON = 24

# 과거 1주 사용
INPUT_SIZE = 24 * 7

# 학습 설정
MAX_STEPS = 200
BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
WINDOWS_BATCH_SIZE = 256
INFERENCE_WINDOWS_BATCH_SIZE = 256

# GPU 설정
USE_GPU = torch.cuda.is_available()
ACCELERATOR = "gpu" if USE_GPU else "cpu"
DEVICES = 1

# DeepAR 안정성을 위해 FP32 사용
PRECISION = "32-true"

# 그래프에 학습구간 마지막 30일 표시
PLOT_CONTEXT_HOURS = 24 * 30


# ------------------------------------------------------------
# 3) Data loading / preprocessing
# ------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """
    원본 데이터는 같은 region 안에 여러 센터(no)가 있으므로,
    region별 시간당 전력 사용량 합계로 집계해서 region-level 시계열 생성.
    최종 컬럼: unique_id, ds, y
    """
    df = pd.read_csv(csv_path)

    required_cols = {"no", "region", "year", "month", "day", "hour", "elec_kwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    # 원본 hour는 1~24 이므로 1을 빼서 0~23시로 맞춤
    df["ds"] = (
        pd.to_datetime(df[["year", "month", "day"]])
        + pd.to_timedelta(df["hour"] - 1, unit="h")
    )

    # region 단위 합산
    ts = (
        df.groupby(["region", "ds"], as_index=False)["elec_kwh"]
        .sum()
        .rename(columns={"region": "unique_id", "elec_kwh": "y"})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )

    return ts


def repair_missing_hours(panel_df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    """
    지역별로 전체 시간축에 맞춰 reindex.
    빠진 시간은 시간기반 보간 + 앞뒤 채움.
    """
    repaired = []

    global_start = panel_df["ds"].min()
    global_end = panel_df["ds"].max()
    full_index = pd.date_range(global_start, global_end, freq=freq)

    for uid, g in panel_df.groupby("unique_id"):
        g = g.sort_values("ds").set_index("ds")[["y"]]
        g = g.reindex(full_index)
        g["unique_id"] = uid

        missing_before = g["y"].isna().sum()
        if missing_before > 0:
            print(f"[INFO] {uid}: missing {missing_before} hours -> interpolated")

        g["y"] = g["y"].interpolate(method="time").bfill().ffill()
        g = g.reset_index().rename(columns={"index": "ds"})
        repaired.append(g[["unique_id", "ds", "y"]])

    out = (
        pd.concat(repaired, axis=0)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return out


def check_continuity(panel_df: pd.DataFrame, freq: str = "H"):
    broken = []

    for uid, g in panel_df.groupby("unique_id"):
        g = g.sort_values("ds")
        expected = pd.date_range(g["ds"].min(), g["ds"].max(), freq=freq)
        actual = pd.DatetimeIndex(g["ds"])

        if len(expected) != len(actual) or not expected.equals(actual):
            broken.append(uid)

    if broken:
        print(f"[WARN] 시간축 누락/불연속 지역: {broken}")
    else:
        print("[OK] 모든 지역의 시간축이 연속적입니다.")


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


def split_train_valid(panel_df: pd.DataFrame, val_start: pd.Timestamp):
    train_df = panel_df[panel_df["ds"] < val_start].copy()
    valid_df = panel_df[panel_df["ds"] >= val_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("train 또는 validation 데이터가 비어 있습니다.")

    val_counts = valid_df.groupby("unique_id")["ds"].count()
    if val_counts.nunique() != 1:
        raise ValueError(f"지역별 validation 길이가 다릅니다.\n{val_counts}")

    valid_hours = int(val_counts.iloc[0])

    print(f"Train period : {train_df['ds'].min()} ~ {train_df['ds'].max()}")
    print(f"Valid period : {valid_df['ds'].min()} ~ {valid_df['ds'].max()}")
    print(f"Validation   : {valid_hours} hours")
    print(f"Pred horizon : {PRED_HORIZON} hours per step")
    print(f"#Regions     : {panel_df['unique_id'].nunique()}")

    return train_df, valid_df, valid_hours


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


def evaluate_predictions(valid_df: pd.DataFrame, pred_df: pd.DataFrame, model_names):
    merged = valid_df.merge(pred_df, on=["unique_id", "ds"], how="inner")

    overall_rows = []
    by_region_rows = []

    for model in model_names:
        y_true = merged["y"].values
        y_pred = merged[model].values

        overall_rows.append({
            "model": model,
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE(%)": mape(y_true, y_pred),
            "sMAPE(%)": smape(y_true, y_pred),
        })

        for uid, g in merged.groupby("unique_id"):
            yt = g["y"].values
            yp = g[model].values
            by_region_rows.append({
                "unique_id": uid,
                "model": model,
                "MAE": mae(yt, yp),
                "RMSE": rmse(yt, yp),
                "MAPE(%)": mape(yt, yp),
                "sMAPE(%)": smape(yt, yp),
            })

    overall_df = (
        pd.DataFrame(overall_rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )
    by_region_df = (
        pd.DataFrame(by_region_rows)
        .sort_values(["unique_id", "RMSE"])
        .reset_index(drop=True)
    )

    return merged, overall_df, by_region_df


# ------------------------------------------------------------
# 5) Model builder
# ------------------------------------------------------------
def build_models(futr_exog_list):
    trainer_kwargs = {
        "accelerator": ACCELERATOR,
        "devices": DEVICES,
        "precision": PRECISION,
        "enable_progress_bar": True,
        "enable_model_summary": True,
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
# 6) Rolling forecast
# ------------------------------------------------------------
def rolling_forecast(nf, train_nf, valid_nf, futr_exog_list):
    """
    validation 6개월을 PRED_HORIZON씩 끊어서 순차 예측.
    actual 값을 매 스텝 history에 추가하는 방식.
    """
    history_df = train_nf.copy()
    pred_parts = []

    valid_start = valid_nf["ds"].min()
    valid_end = valid_nf["ds"].max()

    cutoff_points = pd.date_range(valid_start, valid_end, freq=f"{PRED_HORIZON}H")

    for i, cutoff in enumerate(cutoff_points, 1):
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

        if i % 20 == 0 or i == 1 or i == len(cutoff_points):
            print(f"[INFO] Rolling forecast step {i}/{len(cutoff_points)} done")

    pred_df = pd.concat(pred_parts, axis=0, ignore_index=True)
    pred_df = pred_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return pred_df


# ------------------------------------------------------------
# 7) Plotting
# ------------------------------------------------------------
def save_region_plots(train_df, valid_df, pred_df, model_names, plot_dir: Path):
    merged = valid_df.merge(pred_df, on=["unique_id", "ds"], how="inner")

    for uid in merged["unique_id"].unique():
        train_g = train_df[train_df["unique_id"] == uid].copy()
        valid_g = merged[merged["unique_id"] == uid].copy()

        context_g = train_g.tail(PLOT_CONTEXT_HOURS)

        plt.figure(figsize=(16, 6))
        plt.plot(context_g["ds"], context_g["y"], label="train_recent_actual")
        plt.plot(valid_g["ds"], valid_g["y"], label="valid_actual")

        for model in model_names:
            plt.plot(valid_g["ds"], valid_g[model], label=model)

        plt.title(f"Region: {uid} | Validation Forecast")
        plt.xlabel("datetime")
        plt.ylabel("elec_kwh")
        plt.legend()
        plt.tight_layout()

        out_path = plot_dir / f"{uid}_validation_forecast.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def save_region_plots_per_model(valid_df, pred_df, model_names, plot_dir: Path):
    merged = valid_df.merge(pred_df, on=["unique_id", "ds"], how="inner")

    for uid in merged["unique_id"].unique():
        uid_df = merged[merged["unique_id"] == uid].copy()

        for model in model_names:
            plt.figure(figsize=(16, 5))
            plt.plot(uid_df["ds"], uid_df["y"], label="actual")
            plt.plot(uid_df["ds"], uid_df[model], label=model)
            plt.title(f"Region: {uid} | {model}")
            plt.xlabel("datetime")
            plt.ylabel("elec_kwh")
            plt.legend()
            plt.tight_layout()

            out_path = plot_dir / f"{uid}_{model}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()


# ------------------------------------------------------------
# 8) Main
# ------------------------------------------------------------
def main():
    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if USE_GPU:
        print(f"[INFO] GPU name = {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] GPU not found. Running on CPU.")

    print("[INFO] Loading data...")
    panel = load_data(CSV_PATH)

    print("[INFO] Aggregated panel preview:")
    print(panel.head())

    print("[INFO] Rows by region before repair:")
    print(panel.groupby("unique_id").size())

    check_continuity(panel, freq=FREQ)

    print("[INFO] Repairing missing hours...")
    panel = repair_missing_hours(panel, freq=FREQ)

    print("[INFO] Rows by region after repair:")
    print(panel.groupby("unique_id").size())

    check_continuity(panel, freq=FREQ)

    print("[INFO] Creating time features...")
    panel = add_time_features(panel)

    print("[INFO] Splitting train/validation...")
    train_df, valid_df, valid_hours = split_train_valid(panel, VAL_START)

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

    train_nf = train_df[["unique_id", "ds", "y"] + futr_exog_list].copy()
    valid_nf = valid_df[["unique_id", "ds", "y"] + futr_exog_list].copy()

    print("[INFO] Building models...")
    models = build_models(futr_exog_list=futr_exog_list)
    model_names = [type(m).__name__ for m in models]
    print(f"[INFO] Models = {model_names}")

    nf = NeuralForecast(models=models, freq=FREQ)

    print("[INFO] Training started...")
    nf.fit(df=train_nf)

    print("[INFO] Rolling forecasting validation horizon...")
    pred_df = rolling_forecast(
        nf=nf,
        train_nf=train_nf,
        valid_nf=valid_nf,
        futr_exog_list=futr_exog_list,
    )

    print("[INFO] Evaluating...")
    merged_pred, overall_df, by_region_df = evaluate_predictions(
        valid_df=valid_nf[["unique_id", "ds", "y"]],
        pred_df=pred_df,
        model_names=model_names,
    )

    pred_path = OUT_DIR / "validation_predictions.csv"
    overall_path = OUT_DIR / "validation_metrics_overall.csv"
    by_region_path = OUT_DIR / "validation_metrics_by_region.csv"

    merged_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    by_region_df.to_csv(by_region_path, index=False, encoding="utf-8-sig")

    print("\n===== Overall Validation Metrics =====")
    print(overall_df)

    print("[INFO] Saving region plots...")
    save_region_plots(
        train_df=train_df[["unique_id", "ds", "y"]],
        valid_df=valid_nf[["unique_id", "ds", "y"]],
        pred_df=pred_df,
        model_names=model_names,
        plot_dir=PLOT_DIR,
    )

    print("[INFO] Saving per-model region plots...")
    save_region_plots_per_model(
        valid_df=valid_nf[["unique_id", "ds", "y"]],
        pred_df=pred_df,
        model_names=model_names,
        plot_dir=PER_MODEL_PLOT_DIR,
    )

    print("\n[INFO] Done.")
    print(f"Saved predictions : {pred_path}")
    print(f"Saved overall     : {overall_path}")
    print(f"Saved by region   : {by_region_path}")
    print(f"Saved plots dir   : {PLOT_DIR}")


if __name__ == "__main__":
    main()