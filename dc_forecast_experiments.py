#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터센터 전력수요 예측 — 실험 E0~E3, E8  (v1.2 패치)

v1.2 변경점
  [PATCH-1] E8: center_mature_util 의 np.nan 가드 버그 수정(파이썬에서 `not nan`=False),
            성숙 센터는 '말기 6개월(여름 포함)'이 아니라 '연평균 가동률'을 사용해 계절 과대예측 제거.
  [PATCH-2] E3: 성숙수준 앵커를 노이즈 큰 5-NN 평균 대신 '용량대(capacity-band) 연평균 가동률 중앙값'으로,
            관측 6개월 미만 신규센터 제외, (a)트리비얼 베이스라인을 명시적 '넘어야 할 기준선'으로 보고,
            평균과 함께 중앙값·승률((c)<(a),(c)<(b))을 같이 출력.
            (b)와 (c)가 동일 앵커를 공유하도록 해 ramp의 순수 기여를 분리(ablation).
  [PATCH-3] 인코딩 자동감지 순서를 utf-8-sig(BOM) 우선으로.

필요 패키지: pandas numpy scipy scikit-learn matplotlib
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
LOAD_CSV   = "data.csv"
META_CSV   = "dc_customers_2025_utf8_bom.csv"
OUTPUT_DIR = "exp_out"

WINDOW_START = pd.Timestamp("2020-06-01")
WINDOW_END   = pd.Timestamp("2023-12-31")
HOURS_2023   = 8760                       # 비윤년
HOURS_2024   = 8784                       # 윤년 -> 계약전력(연) 컬럼 기준
MATURE_AGE_M = 36                          # 이보다 오래된 센터는 성숙으로 간주
MIN_MONTHS   = 6                           # E3에서 평가할 신규센터 최소 관측월
MAKE_PLOTS   = True

# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------
def read_csv_auto(path, usecols=None):
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):   # [PATCH-3] BOM 우선
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols)
        except Exception as e:
            last = e
    raise last


def to_num(series):
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
    for c in cols:
        if all(k in str(c) for k in keywords):
            return c
    return None


def is_pos(x):
    """[PATCH-1] nan/inf/<=0 을 한 번에 거르는 안전 체크"""
    try:
        return (x is not None) and np.isfinite(x) and (x > 0)
    except Exception:
        return False


SUDOGWON = {"서울", "경기", "인천"}

# ============================================================================
# E0
# ============================================================================
def load_metadata():
    path = META_CSV or sorted(glob.glob("dc_customers_2025_*.csv"))[0]
    m = read_csv_auto(path)
    cols = list(m.columns)
    c_no   = find_col(cols, "번호") or cols[0]
    c_reg  = find_col(cols, "지역")
    c_op   = find_col(cols, "고객명")
    cap_cands = [c for c in cols if str(c).startswith("계약전력") and "연" not in str(c)]
    c_cap  = cap_cands[0] if cap_cands else find_col(cols, "계약전력")
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
    try:
        load = read_csv_auto(LOAD_CSV, usecols=["no", "year", "month", "elec_kwh"])
    except Exception:
        load = read_csv_auto(LOAD_CSV)
        load = load.rename(columns={c: str(c).strip().lower() for c in load.columns})
    load["elec_kwh"] = to_num(load["elec_kwh"])
    g = load.groupby(["no", "year", "month"], as_index=False)["elec_kwh"].sum()
    g["ym"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
    g["hours_in_month"] = g["ym"].dt.daysinmonth * 24
    panel = g.merge(meta, on="no", how="left")
    matched = panel["cap_kw"].notna().mean()
    print(f"[E0] 부하 로드: {LOAD_CSV} | 센터 {g.no.nunique()}개, "
          f"월 관측 {len(g)}행 | 메타 조인 매칭율 {matched:.1%}")
    panel["util"] = panel["elec_kwh"] / (panel["cap_kw"] * panel["hours_in_month"])
    panel["age"] = ((panel["ym"].dt.year - panel["energized"].dt.year) * 12
                    + (panel["ym"].dt.month - panel["energized"].dt.month))
    return panel


def center_mature_util(panel, no, year_max=2023, last_k=6):
    d = panel[(panel.no == no) & (panel.ym.dt.year <= year_max)].sort_values("ym")
    d = d[d.util.notna() & (d.util > 0)]
    if len(d) == 0:
        return np.nan
    return d.tail(last_k).util.mean()


def recent_util(panel, no, k=3):
    """[PATCH] 최근 k개월 평균 가동률 — 신규센터 현재 ramp 수준 추정"""
    d = panel[(panel.no == no) & (panel.util.notna()) & (panel.util > 0)].sort_values("ym")
    if len(d) == 0:
        return np.nan
    return d.tail(k).util.mean()


def annual_util_2023(panel, meta):
    """[PATCH] 센터별 2023 '연평균' 가동률 (계절편향 없는 수준 추정). Series(no->util)"""
    ann = panel[panel.ym.dt.year == 2023].groupby("no")["elec_kwh"].sum()
    cap = meta.set_index("no")["cap_kw"]
    u = (ann / (cap * HOURS_2023)).replace([np.inf, -np.inf], np.nan)
    return u


# ============================================================================
# E1
# ============================================================================
def e1_descriptive(panel, meta):
    print("\n=== [E1] 이질성 기술통계 ===")
    util23 = annual_util_2023(panel, meta).dropna()
    print("계약전력(kW) 분포:",
          meta.cap_kw.describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_dict())
    print("2023 연평균 가동률 분포:",
          util23.describe()[["min", "25%", "50%", "75%", "max"]].round(3).to_dict())
    print("수도권/비수도권:",
          dict(meta.sudogwon.map({1: "수도권", 0: "비수도권"}).value_counts()))
    print("상위 사업자:", dict(meta.operator.value_counts().head(6)))
    if MAKE_PLOTS:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(np.log10(meta.cap_kw.dropna()), bins=20, color="#4C78A8")
        ax[0].set_title("Contract Capacity Distribution"); ax[0].set_xlabel("log10 kW")
        ax[1].hist(util23.clip(0, 1.2), bins=20, color="#F58518")
        ax[1].set_title("2023 Annual Utilization"); ax[1].set_xlabel("util = annual energy / (cap * 8760)")
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "E1_heterogeneity.png"), dpi=130)
        plt.close(fig); print("  -> 그림 저장: E1_heterogeneity.png")
    return util23


# ============================================================================
# E2
# ============================================================================
def logistic(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))


def fit_ramp(panel, new_ids):
    from scipy.optimize import curve_fit
    rows = []
    for no in new_ids:
        mu = center_mature_util(panel, no)
        if not is_pos(mu):                       # [PATCH-1] 안전 가드
            continue
        d = panel[(panel.no == no) & (panel.age >= 0) & panel.util.notna()]
        for _, r in d.iterrows():
            rows.append((r["age"], min(r["util"] / mu, 1.5)))
    pts = pd.DataFrame(rows, columns=["age", "u_norm"])
    if len(pts) < 6:
        return None, pts
    try:
        popt, _ = curve_fit(logistic, pts.age, pts.u_norm, p0=[1.0, 0.2, 6.0],
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
        print("  (점 부족으로 적합 실패)"); return None
    L, k, t0 = popt
    print(f"  로지스틱 적합: L={L:.3f}, k={k:.3f}, t0={t0:.1f}개월  (점 {len(pts)}개)")
    if MAKE_PLOTS:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(pts.age, pts.u_norm, s=14, alpha=0.5, label="New center observations")
        xs = np.linspace(0, max(pts.age.max(), 36), 100)
        ax.plot(xs, logistic(xs, *popt), "r-", lw=2, label="Logistic fit u(age)")
        ax.set_xlabel("Months Since Energization (age)"); ax.set_ylabel("Normalized Utilization (mature=1)")
        ax.set_title("Ramp Curve"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "E2_ramp_curve.png"), dpi=130)
        plt.close(fig); print("  -> 그림 저장: E2_ramp_curve.png")
    return popt


# ----------------------------------------------------------------------------
# [PATCH-2] 성숙수준 앵커: 용량대(capacity-band) 연평균 가동률 중앙값
# ----------------------------------------------------------------------------
def capacity_band_anchor(meta, util_ann, target_no, exclude, ratio=2.0):
    t_cap = meta.loc[meta.no == target_no, "cap_kw"].iloc[0]
    mature = meta[(~meta.is_new) & (~meta.no.isin(exclude)) & meta.cap_kw.notna()]
    band = mature[(mature.cap_kw >= t_cap / ratio) & (mature.cap_kw <= t_cap * ratio)]
    vals = util_ann.reindex(band.no).dropna()
    if len(vals) < 3:
        vals = util_ann.reindex(mature.no).dropna()
    return float(vals.median()) if len(vals) else np.nan


def wape(actual, pred):
    actual = np.asarray(actual, float); pred = np.asarray(pred, float)
    m = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[m], pred[m]
    s = np.abs(actual).sum()
    return np.abs(actual - pred).sum() / s if s > 0 else np.nan


# ============================================================================
# E3  (PATCH-2)
# ============================================================================
def e3_loo(panel, meta, global_ramp):
    print("\n=== [E3] Leave-one-center-out cold-start (신규센터) ===")
    util_ann = annual_util_2023(panel, meta)
    global_anchor = float(util_ann.reindex(meta.loc[~meta.is_new, "no"]).dropna().median())
    new_ids = meta.loc[meta.is_new, "no"].tolist()
    res = []
    for h in new_ids:
        actual = (panel[(panel.no == h) & (panel.age >= 0) & panel.util.notna()].sort_values("age"))
        if len(actual) < MIN_MONTHS:                   # [PATCH-2] 관측 6개월 미만 제외
            continue
        cap_h = meta.loc[meta.no == h, "cap_kw"].iloc[0]
        hours, ages, y = actual.hours_in_month.values, actual.age.values, actual.elec_kwh.values

        anchor = capacity_band_anchor(meta, util_ann, h, exclude=[h])
        if not is_pos(anchor):
            anchor = global_anchor
        popt_loo, _ = fit_ramp(panel, [n for n in new_ids if n != h])
        ramp = popt_loo if popt_loo is not None else global_ramp

        pred_a = global_anchor * cap_h * hours                       # (a) 트리비얼 상수
        pred_b = anchor * cap_h * hours                              # (b) 용량대 앵커 상수
        pred_c = logistic(ages, *ramp) * anchor * cap_h * hours      # (c) 앵커 x ramp

        res.append({"no": h, "n": len(actual),
                    "WAPE_a_trivial": wape(y, pred_a),
                    "WAPE_b_anchor":  wape(y, pred_b),
                    "WAPE_c_ramp":    wape(y, pred_c)})
    df = pd.DataFrame(res)
    if not len(df):
        print("  (평가 가능한 신규센터 없음 — MIN_MONTHS 완화 필요)"); return df
    print(df.round(3).to_string(index=False))
    print(f"\n  평가 센터 {len(df)}개 (관측 {MIN_MONTHS}개월 이상)")
    print("  평균 WAPE  | (a)트리비얼:%.3f  (b)앵커:%.3f  (c)앵커+ramp:%.3f"
          % (df.WAPE_a_trivial.mean(), df.WAPE_b_anchor.mean(), df.WAPE_c_ramp.mean()))
    print("  중앙 WAPE  | (a)트리비얼:%.3f  (b)앵커:%.3f  (c)앵커+ramp:%.3f"
          % (df.WAPE_a_trivial.median(), df.WAPE_b_anchor.median(), df.WAPE_c_ramp.median()))
    print("  승률  (c)<(b):%.0f%%   (c)<(a):%.0f%%   <- (a)를 이겨야 '트리비얼 초과' 입증"
          % ((df.WAPE_c_ramp < df.WAPE_b_anchor).mean()*100,
             (df.WAPE_c_ramp < df.WAPE_a_trivial).mean()*100))
    df.to_csv(os.path.join(OUTPUT_DIR, "E3_loo_results.csv"), index=False)
    return df


# ============================================================================
# E8  (PATCH-1)
# ============================================================================
def e8_validate_2024(panel, meta, global_ramp):
    print("\n=== [E8] 2024 실측 대조 (out-of-sample 연간 검증) ===")
    util_ann = annual_util_2023(panel, meta)
    age_mid2024 = lambda en: np.nan if pd.isna(en) else (2024 - en.year) * 12 + (7 - en.month)
    rows = []
    for _, m in meta.iterrows():
        no, cap, en, target = m.no, m.cap_kw, m.energized, m.use_2024
        if not is_pos(cap) or not is_pos(target):
            continue
        ann23 = panel[(panel.no == no) & (panel.ym.dt.year == 2023)]["elec_kwh"].sum()
        u_ann = util_ann.get(no, np.nan)
        age_end23 = np.nan if pd.isna(en) else (2023 - en.year) * 12 + (12 - en.month)

        if pd.isna(age_end23) or age_end23 > MATURE_AGE_M:
            # [PATCH-1] 성숙센터: 연평균 가동률 사용(여름편향 제거)
            u2024 = u_ann if is_pos(u_ann) else (ann23 / (cap * HOURS_2023))
        else:
            recent = recent_util(panel, no, k=3)       # 신규/ramp중: 최근 수준을 ramp로 외삽
            if not is_pos(recent):
                recent = u_ann
            frac_now = max(logistic(age_end23, *global_ramp), 0.15)
            plateau  = recent / frac_now
            u2024    = min(plateau * logistic(age_mid2024(en), *global_ramp), 1.2)
        if not is_pos(u2024):                          # [PATCH-1] 최종 nan 가드
            continue
        rows.append({"no": no, "is_new": bool(m.is_new), "target_2024": target,
                     "pred_ours": cap * HOURS_2024 * u2024, "pred_naive": ann23})
    df = pd.DataFrame(rows)
    if not len(df):
        print("  (대조 가능 센터 없음)"); return df
    print(f"  대조 센터 {len(df)}개")
    print("  전체   WAPE | 우리:%.3f  무성장:%.3f"
          % (wape(df.target_2024, df.pred_ours), wape(df.target_2024, df.pred_naive)))
    for label, sub in [("신규", df[df.is_new]), ("성숙", df[~df.is_new])]:
        if len(sub):
            print("  %s WAPE | 우리:%.3f  무성장:%.3f  (%d개)"
                  % (label, wape(sub.target_2024, sub.pred_ours),
                     wape(sub.target_2024, sub.pred_naive), len(sub)))
    df.to_csv(os.path.join(OUTPUT_DIR, "E8_validate_2024.csv"), index=False)
    if MAKE_PLOTS:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(df.target_2024, df.pred_ours, s=18, alpha=0.6, label="Proposed")
        lim = [0, float(df[["target_2024", "pred_ours"]].values.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="y=x")
        ax.set_xlabel("Actual 2024 Annual kWh"); ax.set_ylabel("Predicted 2024 Annual kWh")
        ax.set_title("E8: 2024 Prediction vs Actual"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "E8_pred_vs_actual.png"), dpi=130)
        plt.close(fig); print("  -> 그림 저장: E8_pred_vs_actual.png")
    return df


# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70); print("DC 전력수요 예측 실험 (E0~E3, E8)  v1.2"); print("=" * 70)
    meta  = load_metadata()
    panel = load_monthly_panel(meta)
    e1_descriptive(panel, meta)
    ramp  = e2_ramp(panel, meta)
    if ramp is None:
        ramp = np.array([1.0, 0.2, 6.0])
    e3_loo(panel, meta, ramp)
    e8_validate_2024(panel, meta, ramp)
    print("\n완료. 결과/그림은 '%s/' 폴더 확인." % OUTPUT_DIR)


if __name__ == "__main__":
    main()
