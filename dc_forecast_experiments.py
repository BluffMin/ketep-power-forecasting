#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터센터 전력수요 예측 — 부하데이터 + 메타데이터만으로 돌리는 핵심 실험 (E0~E3, E8)

다루는 실험
  E0  정합·전처리 (조인, 파싱, 월별 집계, 신규센터 식별)
  E1  이질성 기술통계 (계약전력·가동률·사업자 분포)            -> 스케일보정/MoE 필요성 근거
  E2  Ramp 곡선 추정 (신규센터 age 정렬 + 로지스틱 적합)        -> 청구항5, 도4
  E3  Leave-one-center-out cold-start 비교 (a/b/c 3방식)        -> 청구항8, 표1
  E8  2024 실측 대조 (out-of-sample 연간 검증)                  -> 본 목표(월/연 예측) 직접 검증

필요 패키지: pandas, numpy, scipy, scikit-learn, matplotlib  (모두 표준 ML 스택)
  설치 예: pip install pandas numpy scipy scikit-learn matplotlib

사용법
  1) 아래 CONFIG의 LOAD_CSV / META_CSV 경로만 본인 환경에 맞게 수정.
  2) python dc_forecast_experiments.py
  3) 콘솔 출력 + OUTPUT_DIR 에 그림/요약 CSV 저장.
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG  -- 여기만 본인 환경에 맞게 수정하세요
# ----------------------------------------------------------------------------
LOAD_CSV   = "data.csv"          # 시간별 부하 CSV (no,region,year,month,day,hour,elec_kwh)
META_CSV   = "dc_customers_2025_utf8_bom.csv"                          # 비우면 dc_customers_2025_*.csv 자동 탐색
OUTPUT_DIR = "exp_out"

WINDOW_START = pd.Timestamp("2020-06-01")
WINDOW_END   = pd.Timestamp("2023-12-31")
HOURS_2024   = 8784                      # 2024는 윤년 -> 계약전력(연) 컬럼과 동일 기준

KNN_K        = 5                          # E3/E8 유사센터 개수
MAKE_PLOTS   = True

# ----------------------------------------------------------------------------
# 유틸: 인코딩 자동 감지 read_csv
# ----------------------------------------------------------------------------
def read_csv_auto(path, usecols=None):
    last = None
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols)
        except Exception as e:
            last = e
    raise last


def to_num(series):
    """콤마/공백 섞인 숫자 문자열 -> float"""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def parse_korean_date(s):
    if pd.isna(s):
        return pd.NaT
    nums = re.findall(r"\d+", str(s))
    if len(nums) >= 3:
        try:
            return pd.Timestamp(int(nums[0]), int(nums[1]), int(nums[2]))
        except Exception:
            return pd.NaT
    return pd.NaT


def find_col(cols, *keywords):
    """부분일치로 컬럼명 찾기 (메타 컬럼명이 환경마다 미세하게 다를 수 있어 방어)"""
    for c in cols:
        if all(k in str(c) for k in keywords):
            return c
    return None


SUDOGWON = {"서울", "경기", "인천"}

# ============================================================================
# E0  정합 · 전처리
# ============================================================================
def load_metadata():
    path = META_CSV or sorted(glob.glob("dc_customers_2025_*.csv"))[0]
    m = read_csv_auto(path)
    cols = list(m.columns)
    c_no   = find_col(cols, "번호") or cols[0]
    c_reg  = find_col(cols, "지역")
    c_op   = find_col(cols, "고객명")
    c_cap  = find_col(cols, "계약전력") and [c for c in cols if c.startswith("계약전력") and "연" not in c][0]
    c_use24= find_col(cols, "24", "사용량") or find_col(cols, "전력사용량")
    c_date = find_col(cols, "송전일자")
    c_addr = find_col(cols, "주소")

    meta = pd.DataFrame({
        "no":        m[c_no].astype(int),
        "region":    m[c_reg] if c_reg else np.nan,
        "operator":  m[c_op] if c_op else "UNK",
        "cap_kw":    to_num(m[c_cap]),
        "use_2024":  to_num(m[c_use24]) if c_use24 else np.nan,
        "energized": m[c_date].apply(parse_korean_date) if c_date else pd.NaT,
        "address":   m[c_addr] if c_addr else "",
    })
    meta["sudogwon"] = meta["region"].isin(SUDOGWON).astype(int)
    meta["is_new"] = meta["energized"].between(WINDOW_START, WINDOW_END)
    print(f"[E0] 메타 로드: {path}")
    print(f"     센터 {len(meta)}개 | 계약전력 결측 {meta.cap_kw.isna().sum()} | "
          f"송전일자 결측 {meta.energized.isna().sum()} | 기간내 신규 {int(meta.is_new.sum())}개")
    return meta


def load_monthly_panel(meta):
    # 부하 CSV는 월별 집계만 필요 -> 필요한 컬럼만 읽어 메모리 절약
    try:
        load = read_csv_auto(LOAD_CSV, usecols=["no", "year", "month", "elec_kwh"])
    except Exception:
        load = read_csv_auto(LOAD_CSV)
        load = load.rename(columns={c: c.strip().lower() for c in load.columns})
    load["elec_kwh"] = to_num(load["elec_kwh"])
    g = load.groupby(["no", "year", "month"], as_index=False)["elec_kwh"].sum()
    g["ym"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
    g["hours_in_month"] = g["ym"].dt.daysinmonth * 24

    panel = g.merge(meta, on="no", how="left")
    matched = panel["cap_kw"].notna().sum() / len(panel)
    print(f"[E0] 부하 로드: {LOAD_CSV} | 센터 {g.no.nunique()}개, "
          f"월 관측 {len(g)}행 | 메타 조인 매칭율 {matched:.1%}")

    # 가동률(util) = 월 사용량 / (계약전력 * 월 시간)
    panel["util"] = panel["elec_kwh"] / (panel["cap_kw"] * panel["hours_in_month"])
    # age(월) = 송전월로부터 경과
    panel["age"] = ((panel["ym"].dt.year - panel["energized"].dt.year) * 12
                    + (panel["ym"].dt.month - panel["energized"].dt.month))
    return panel


def center_mature_util(panel, no, year_max=2023, last_k=6):
    """해당 센터의 '관측 말기' 평균 가동률 (성숙 가동률 추정치)"""
    d = panel[(panel.no == no) & (panel.ym.dt.year <= year_max)].sort_values("ym")
    d = d[d.util.notna() & (d.util > 0)]
    if len(d) == 0:
        return np.nan
    return d.tail(last_k).util.mean()


# ============================================================================
# E1  이질성 기술통계
# ============================================================================
def e1_descriptive(panel, meta):
    print("\n=== [E1] 이질성 기술통계 ===")
    # 2023 기준 가동률 (부하데이터로 재계산: 메타의 달성률은 2024라 시점 다름)
    util23 = (panel[panel.ym.dt.year == 2023]
              .groupby("no")["elec_kwh"].sum()
              / (meta.set_index("no")["cap_kw"] * (365 * 24)))
    util23 = util23.replace([np.inf, -np.inf], np.nan).dropna()
    print("계약전력(kW) 분포:",
          meta.cap_kw.describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_dict())
    print("2023 가동률 분포:",
          util23.describe()[["min", "25%", "50%", "75%", "max"]].round(3).to_dict())
    print("수도권/비수도권:",
          dict(meta.sudogwon.map({1: "수도권", 0: "비수도권"}).value_counts()))
    print("상위 사업자:", dict(meta.operator.value_counts().head(6)))

    if MAKE_PLOTS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(np.log10(meta.cap_kw.dropna()), bins=20, color="#4C78A8")
        ax[0].set_title("Contract Capacity Distribution"); ax[0].set_xlabel("log10 kW")
        ax[1].hist(util23.clip(0, 1.2), bins=20, color="#F58518")
        ax[1].set_title("2023 Utilization Distribution"); ax[1].set_xlabel("util = energy / (capacity * 8760)")
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "E1_heterogeneity.png"), dpi=130)
        plt.close(fig)
        print("  -> 그림 저장: E1_heterogeneity.png")
    return util23


# ============================================================================
# E2  Ramp 곡선 추정
# ============================================================================
def logistic(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))


def fit_ramp(panel, new_ids):
    """신규센터들의 (age, 정규화 가동률) 점들에 로지스틱 적합. 정규화=각 센터의 성숙가동률로 나눔."""
    from scipy.optimize import curve_fit
    rows = []
    for no in new_ids:
        mu = center_mature_util(panel, no)
        if not mu or mu <= 0:
            continue
        d = panel[(panel.no == no) & (panel.age >= 0) & panel.util.notna()]
        for _, r in d.iterrows():
            rows.append((r["age"], min(r["util"] / mu, 1.5)))
    if len(rows) < 6:
        return None, pd.DataFrame(rows, columns=["age", "u_norm"])
    pts = pd.DataFrame(rows, columns=["age", "u_norm"])
    try:
        popt, _ = curve_fit(logistic, pts.age, pts.u_norm,
                            p0=[1.0, 0.2, 6.0],
                            bounds=([0.5, 0.01, -6], [1.5, 2.0, 48]), maxfev=10000)
    except Exception:
        popt = None
    return popt, pts


def e2_ramp(panel, meta):
    print("\n=== [E2] Ramp 곡선 추정 ===")
    new_ids = meta.loc[meta.is_new, "no"].tolist()
    print(f"기간내 신규센터 {len(new_ids)}개 사용:", new_ids)
    popt, pts = fit_ramp(panel, new_ids)
    if popt is None:
        print("  (점 부족으로 적합 실패 — 신규센터/관측이 더 필요)")
        return None
    L, k, t0 = popt
    print(f"  로지스틱 적합: L={L:.3f}, k={k:.3f}, t0={t0:.1f}개월  (점 {len(pts)}개)")
    print(f"  해석: 가동개시 후 약 {t0:.0f}개월에 성숙의 50%, k가 클수록 가파른 ramp")
    if MAKE_PLOTS:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(pts.age, pts.u_norm, s=14, alpha=0.5, label="New center observations")
        xs = np.linspace(0, max(pts.age.max(), 36), 100)
        ax.plot(xs, logistic(xs, *popt), "r-", lw=2, label="Logistic fit u(age)")
        ax.set_xlabel("Months Since Energization (age)"); ax.set_ylabel("Normalized Utilization (mature = 1)")
        ax.set_title("Ramp Curve"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "E2_ramp_curve.png"), dpi=130)
        plt.close(fig); print("  -> 그림 저장: E2_ramp_curve.png")
    return popt


# ----------------------------------------------------------------------------
# 유사센터 (메타 기반 k-NN): 특징 = log(용량), 수도권여부, 사업자일치
# ----------------------------------------------------------------------------
def similar_centers(meta, target_no, exclude, k=KNN_K):
    base = meta[~meta.no.isin(exclude) & meta.cap_kw.notna()].copy()
    t = meta.loc[meta.no == target_no].iloc[0]
    base["d_cap"] = (np.log10(base.cap_kw) - np.log10(t.cap_kw)).abs()
    base["d_reg"] = (base.sudogwon != t.sudogwon).astype(float)
    base["d_op"]  = (base.operator != t.operator).astype(float)
    base["dist"]  = base.d_cap + 0.5 * base.d_reg + 0.3 * base.d_op
    return base.nsmallest(k, "dist")


def wape(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    s = np.abs(actual).sum()
    return np.abs(actual - pred).sum() / s if s > 0 else np.nan


# ============================================================================
# E3  Leave-one-center-out cold-start
# ============================================================================
def e3_loo(panel, meta, global_ramp):
    print("\n=== [E3] Leave-one-center-out cold-start (신규센터) ===")
    new_ids = meta.loc[meta.is_new, "no"].tolist()
    global_util = panel[panel.util.notna() & (panel.util > 0)].util.median()
    res = []
    for h in new_ids:
        actual = (panel[(panel.no == h) & (panel.age >= 0) & panel.util.notna()]
                  .sort_values("age"))
        if len(actual) < 3:
            continue
        cap_h = meta.loc[meta.no == h, "cap_kw"].iloc[0]
        hours = actual.hours_in_month.values
        ages  = actual.age.values
        y     = actual.elec_kwh.values

        sim = similar_centers(meta, h, exclude=[h])
        sim_mu = np.nanmean([center_mature_util(panel, s) for s in sim.no])

        # ramp는 heldout 제외하고 재적합 (정보누수 방지)
        popt_loo, _ = fit_ramp(panel, [n for n in new_ids if n != h])
        ramp = popt_loo if popt_loo is not None else global_ramp

        # (a) 전체중앙값 가동률 (상수)
        pred_a = global_util * cap_h * hours
        # (b) 유사센터 현재(성숙)부하 차용 (상수, ramp 없음)
        pred_b = sim_mu * cap_h * hours
        # (c) age정렬 ramp * 유사성숙가동률
        pred_c = logistic(ages, *ramp) * sim_mu * cap_h * hours

        res.append({
            "no": h, "n": len(actual),
            "WAPE_a_fleet": wape(y, pred_a),
            "WAPE_b_current": wape(y, pred_b),
            "WAPE_c_ramp": wape(y, pred_c),
        })
    df = pd.DataFrame(res)
    if len(df):
        print(df.round(3).to_string(index=False))
        print("\n  평균 WAPE  | (a)전체:%.3f  (b)현재차용:%.3f  (c)ramp정렬:%.3f"
              % (df.WAPE_a_fleet.mean(), df.WAPE_b_current.mean(), df.WAPE_c_ramp.mean()))
        win = (df.WAPE_c_ramp < df.WAPE_b_current).mean()
        print(f"  (c)가 (b)를 이긴 비율: {win:.0%}  ← 핵심 주장(age정렬>현재차용) 검증")
        df.to_csv(os.path.join(OUTPUT_DIR, "E3_loo_results.csv"), index=False)
    else:
        print("  (검증할 신규센터 관측 부족)")
    return df


# ============================================================================
# E8  2024 실측 대조 (out-of-sample 연간 검증)
# ============================================================================
def e8_validate_2024(panel, meta, global_ramp):
    print("\n=== [E8] 2024 실측 대조 (out-of-sample 연간 검증) ===")
    rows = []
    age_mid2024 = lambda en: np.nan if pd.isna(en) else \
        (2024 - en.year) * 12 + (7 - en.month)
    for _, m in meta.iterrows():
        no, cap, en = m.no, m.cap_kw, m.energized
        target = m.use_2024
        if pd.isna(cap) or pd.isna(target):
            continue
        ann23 = panel[(panel.no == no) & (panel.ym.dt.year == 2023)]["elec_kwh"].sum()
        last_u = center_mature_util(panel, no)          # 2023 말기 가동률
        if not last_u or last_u <= 0:
            continue
        # 우리 방식: ramp로 2024 가동률 외삽
        age_end23 = np.nan if pd.isna(en) else (2023 - en.year) * 12 + (12 - en.month)
        if pd.isna(age_end23) or age_end23 > 36:        # 성숙센터: 평탄 유지
            u2024 = last_u
        else:                                            # 신규/ramp중: 곡선 따라 상승
            frac_now = max(logistic(age_end23, *global_ramp), 0.2)
            plateau  = last_u / frac_now
            u2024    = min(plateau * logistic(age_mid2024(en), *global_ramp), 1.2)
        pred_ours  = cap * HOURS_2024 * u2024
        pred_naive = ann23                               # 베이스라인: 2023과 동일(무성장)
        rows.append({"no": no, "is_new": bool(m.is_new),
                     "target_2024": target, "pred_ours": pred_ours, "pred_naive": pred_naive})
    df = pd.DataFrame(rows)
    if not len(df):
        print("  (대조 가능 센터 없음)"); return df
    print(f"  대조 센터 {len(df)}개")
    print("  전체   WAPE | 우리방식:%.3f  무성장베이스라인:%.3f"
          % (wape(df.target_2024, df.pred_ours), wape(df.target_2024, df.pred_naive)))
    if df.is_new.any():
        dn = df[df.is_new]
        print("  신규만 WAPE | 우리방식:%.3f  무성장베이스라인:%.3f  (신규 %d개)"
              % (wape(dn.target_2024, dn.pred_ours), wape(dn.target_2024, dn.pred_naive), len(dn)))
    df.to_csv(os.path.join(OUTPUT_DIR, "E8_validate_2024.csv"), index=False)
    if MAKE_PLOTS:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(df.target_2024, df.pred_ours, s=18, alpha=0.6, label="Proposed method")
        lim = [0, df[["target_2024", "pred_ours"]].values.max() * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="y=x")
        ax.set_xlabel("Actual 2024 Annual kWh"); ax.set_ylabel("Predicted 2024 Annual kWh")
        ax.set_title("E8: 2024 Prediction vs Actual"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "E8_pred_vs_actual.png"), dpi=130)
        plt.close(fig); print("  -> 그림 저장: E8_pred_vs_actual.png")
    return df


# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("데이터센터 전력수요 예측 실험 (E0~E3, E8)")
    print("=" * 70)
    meta  = load_metadata()
    panel = load_monthly_panel(meta)
    e1_descriptive(panel, meta)
    ramp  = e2_ramp(panel, meta)
    if ramp is None:                       # 적합 실패시 완만한 기본 곡선
        ramp = np.array([1.0, 0.2, 6.0])
    e3_loo(panel, meta, ramp)
    e8_validate_2024(panel, meta, ramp)
    print("\n완료. 결과/그림은 '%s/' 폴더를 확인하세요." % OUTPUT_DIR)


if __name__ == "__main__":
    main()
