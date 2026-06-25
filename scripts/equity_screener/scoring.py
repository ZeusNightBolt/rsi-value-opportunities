import math

import numpy as np
import pandas as pd

from .config import SCORE_COLUMNS, SLEEVE_LABELS

def pct_score(series: pd.Series, lower_is_better: bool) -> pd.Series:
    s = series.astype(float).copy()
    valid = s.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    n = len(valid)
    if n == 0:
        return out
    if n == 1:
        out.loc[valid.index] = 50.0
        return out
    ranks = valid.rank(method="average", ascending=True)
    if lower_is_better:
        scores = 100.0 * (n - ranks) / (n - 1)
    else:
        scores = 100.0 * (ranks - 1) / (n - 1)
    out.loc[valid.index] = scores.clip(0, 100)
    return out

def grade_to_score(grade):
    if grade is None or (isinstance(grade, float) and math.isnan(grade)):
        return np.nan
    g = str(grade).strip().upper()[:1]
    return {"A": 100.0, "B": 75.0, "C": 50.0, "D": 25.0, "F": 0.0}.get(g, np.nan)

def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.copy()
    for col in ["yf_forward_pe", "yf_trailing_pe", "yf_price_to_book", "yf_peg_ratio"]:
        df.loc[df[col].astype(float) <= 0, col] = np.nan

    for col in ["short_pct_float", "from_52w_low_pct", "from_52w_high_pct", "ret_1w_pct", "ret_1m_pct", "ret_3m_pct", "ret_6m_pct", "ret_ytd_pct", "volume_vs_20d"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dolt_value_score" in df.columns:
        df["dolt_value_norm"] = (df["dolt_value_score"].astype(float) * 10.0).where(df["dolt_value_score"].notna())
    else:
        df["dolt_value_norm"] = np.nan
    df["grade_value_norm"] = df["value_grade"].map(grade_to_score)
    df["warehouse_value_score"] = df["dolt_value_norm"].combine_first(df["grade_value_norm"]).fillna(50.0)
    df["forward_pe_score"] = pct_score(df["yf_forward_pe"], lower_is_better=True).fillna(50.0)
    df["trailing_pe_score"] = pct_score(df["yf_trailing_pe"], lower_is_better=True).fillna(50.0)
    df["pb_score"] = pct_score(df["yf_price_to_book"], lower_is_better=True).fillna(50.0)
    df["peg_score"] = pct_score(df["yf_peg_ratio"], lower_is_better=True).fillna(50.0)
    df["composite_value_score"] = (
        0.40 * df["warehouse_value_score"]
        + 0.25 * df["forward_pe_score"]
        + 0.15 * df["trailing_pe_score"]
        + 0.15 * df["pb_score"]
        + 0.05 * df["peg_score"]
    )

    # ── Continuous technical factor scores ──
    # Design rule: a defined factor always gets a 0-100 value for every stock.
    # Eligibility flags only decide tab membership; they no longer zero-out scores.
    for c in [
        "rsi0", "rsi1", "rsi2", "rsi3", "rsi4", "rsi5", "rsi_delta_1", "prior_delta_3_avg", "rsi_accel",
        "ret_6m_pct", "ret_3m_pct", "ret_1m_pct", "ret_1w_pct", "price_vs_sma20_pct", "price_vs_sma50_pct",
        "price_vs_sma200_pct", "volume_vs_20d", "from_52w_low_pct", "from_52w_high_pct",
    ]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    df["sector_ret_1m_median"] = df.groupby("sector")["ret_1m_pct"].transform("median")
    df["sector_ret_3m_median"] = df.groupby("sector")["ret_3m_pct"].transform("median")
    df["peer_lag_1m_pct"] = df["sector_ret_1m_median"] - df["ret_1m_pct"]
    df["peer_lag_3m_pct"] = df["sector_ret_3m_median"] - df["ret_3m_pct"]

    # Percentile anchors used by several sleeves.
    df["rsi_accel_pct"] = pct_score(df["rsi_accel"], lower_is_better=False).fillna(50.0)
    df["rsi_delta_pct"] = pct_score(df["rsi_delta_1"], lower_is_better=False).fillna(50.0)
    df["ret_1w_rank"] = pct_score(df["ret_1w_pct"], lower_is_better=False).fillna(50.0)
    df["ret_1m_rank"] = pct_score(df["ret_1m_pct"], lower_is_better=False).fillna(50.0)
    df["ret_3m_rank"] = pct_score(df["ret_3m_pct"], lower_is_better=False).fillna(50.0)
    df["ret_6m_rank"] = pct_score(df["ret_6m_pct"], lower_is_better=False).fillna(50.0)
    df["volume_rank"] = pct_score(df["volume_vs_20d"].clip(upper=3.0), lower_is_better=False).fillna(50.0)
    df["sector_strength_score"] = pct_score(df["sector_ret_1m_median"], lower_is_better=False).fillna(50.0)
    df["sector_strength_3m_score"] = pct_score(df["sector_ret_3m_median"], lower_is_better=False).fillna(50.0)
    df["peer_lag_score"] = (
        0.65 * pct_score(df["peer_lag_1m_pct"], lower_is_better=False).fillna(50.0)
        + 0.35 * pct_score(df["peer_lag_3m_pct"], lower_is_better=False).fillna(50.0)
    ).clip(0, 100)
    df["peer_rally_laggard_flag"] = ((df["sector_ret_1m_median"] > 3.0) & (df["peer_lag_1m_pct"] > 5.0)).astype(int)

    # Price-distribution / market-profile location: 0 = 52w low, 100 = 52w high.
    low_dist = df["from_52w_low_pct"].clip(lower=0)
    high_gap = (-df["from_52w_high_pct"]).clip(lower=0)
    df["price_position_52w"] = (100.0 * low_dist / (low_dist + high_gap).replace(0, np.nan)).clip(0, 100).fillna(50.0)
    df["accumulation_location_score"] = (100.0 - (df["price_position_52w"] - 22.0).abs() * 3.0).clip(0, 100)
    df["markup_location_score"] = (100.0 - (df["price_position_52w"] - 62.0).abs() * 2.0).clip(0, 100)
    df["breakout_location_score"] = (100.0 - (df["price_position_52w"] - 78.0).abs() * 2.2).clip(0, 100)
    # Distribution/exhaustion risk is continuous, not a binary flag.
    # High 52-week location + hot RSI + positive short-term chase raises risk;
    # this penalizes late wave-5 / distribution setups without blanking scores.
    df["distribution_risk_score"] = (
        0.45 * ((df["price_position_52w"] - 75.0).clip(lower=0) / 25.0 * 100.0)
        + 0.35 * ((df["rsi0"] - 60.0).clip(lower=0) / 25.0 * 100.0)
        + 0.20 * (df["ret_1w_pct"].clip(lower=0, upper=12.0) / 12.0 * 100.0)
    ).clip(0, 100).fillna(0.0)

    # Trend/support geometry.
    df["sma_proximity_score"] = (
        0.55 * pct_score(df["price_vs_sma50_pct"].abs(), lower_is_better=True).fillna(50.0)
        + 0.30 * pct_score(df["price_vs_sma200_pct"].abs(), lower_is_better=True).fillna(50.0)
        + 0.15 * pct_score(df["price_vs_sma20_pct"].abs(), lower_is_better=True).fillna(50.0)
    ).clip(0, 100)
    df["trend_alignment_score"] = (
        0.40 * pct_score(df["price_vs_sma50_pct"], lower_is_better=False).fillna(50.0)
        + 0.35 * pct_score(df["price_vs_sma200_pct"], lower_is_better=False).fillna(50.0)
        + 0.25 * pct_score(df["price_vs_sma20_pct"], lower_is_better=False).fillna(50.0)
    ).clip(0, 100)

    # RSI inflection + value: accumulation/base entries get credit even when the turn is early.
    df["rsi_zone_inflection_score"] = (100.0 - (df["rsi0"] - 43.0).abs() * 3.3).clip(0, 100).fillna(50.0)
    grind_turn = (
        (df["rsi_delta_1"].clip(lower=0, upper=8) / 8.0 * 55.0)
        + ((-df["prior_delta_3_avg"]).clip(lower=0, upper=6) / 6.0 * 25.0)
        + (df["rsi_accel"].clip(lower=0, upper=10) / 10.0 * 20.0)
    ).clip(0, 100)
    df["grind_bonus"] = (grind_turn / 100.0 * 12.0).fillna(0.0)
    df["rsi_acceleration_score"] = (
        0.38 * df["rsi_accel_pct"]
        + 0.26 * df["rsi_delta_pct"]
        + 0.18 * df["rsi_zone_inflection_score"]
        + 0.10 * df["accumulation_location_score"]
        + 0.08 * df["volume_rank"]
        + df["grind_bonus"]
    ).clip(0, 100)
    df["rsi_value_score"] = (0.58 * df["rsi_acceleration_score"] + 0.42 * df["composite_value_score"]).clip(0, 100)

    # Squeeze/value sleeves: continuous; no blank/zero undefined cells.
    df["short_score"] = pct_score(df["short_pct_float"], lower_is_better=False).fillna(35.0)
    df["near_low_score"] = pct_score(df["from_52w_low_pct"], lower_is_better=True).fillna(50.0)
    df["oversold_not_broken_score"] = (100.0 - (df["rsi0"] - 42.0).abs() * 2.3).clip(20, 100).fillna(50.0)
    df["squeeze_laggard_score"] = (
        0.35 * df["short_score"]
        + 0.25 * df["near_low_score"]
        + 0.22 * df["peer_lag_score"]
        + 0.10 * df["oversold_not_broken_score"]
        + 0.08 * df["volume_rank"]
        + np.where((df["peer_rally_laggard_flag"] == 1) & (df["short_pct_float"] >= 5), 8.0, 0.0)
    ).clip(0, 100)
    df["value_laggard_score"] = (
        0.42 * df["composite_value_score"]
        + 0.28 * df["peer_lag_score"]
        + 0.18 * df["near_low_score"]
        + 0.12 * df["rsi_acceleration_score"]
    ).clip(0, 100)

    # Momentum leader / wave-3 sleeve: leaders can be extended, but exhaustion is penalized.
    df["rsi_leader_zone_score"] = (100.0 - (df["rsi0"] - 58.0).abs() * 3.0).clip(0, 100).fillna(50.0)
    df["momentum_leader_score"] = (
        0.27 * df["ret_6m_rank"]
        + 0.24 * df["ret_3m_rank"]
        + 0.18 * df["trend_alignment_score"]
        + 0.13 * df["markup_location_score"]
        + 0.10 * df["volume_rank"]
        + 0.08 * df["rsi_leader_zone_score"]
        - 0.12 * df["distribution_risk_score"]
    ).clip(0, 100)
    df["momentum_leader_eligible"] = (
        (df["momentum_leader_score"] >= 62)
        & (df["ret_3m_rank"] >= 55)
        & (df["trend_alignment_score"] >= 50)
        & (df["volume_vs_20d"].fillna(0) >= 0.55)
    )

    # Momentum pullback / wave-2 or wave-4: strong leader, short pullback, RSI reset, support proximity.
    df["pullback_depth"] = (-df["ret_1w_pct"]).clip(lower=0, upper=18)
    df["pullback_score"] = pct_score(df["pullback_depth"], lower_is_better=False).fillna(50.0)
    # Continuous pullback presence: 100 for modest negative week, decays for chase strength,
    # and avoids a 100/0 discontinuity around exactly 0% one-week return.
    df["pullback_presence_score"] = (55.0 - df["ret_1w_pct"].clip(lower=-12, upper=10) * 5.0).clip(0, 100).fillna(50.0)
    df["rsi_cool_score"] = (100.0 - (df["rsi0"] - 46.0).abs() * 3.2).clip(0, 100).fillna(50.0)
    df["momentum_pullback_score"] = (
        0.28 * df["momentum_leader_score"]
        + 0.20 * df["pullback_score"]
        + 0.17 * df["pullback_presence_score"]
        + 0.16 * df["sma_proximity_score"]
        + 0.11 * df["rsi_cool_score"]
        + 0.08 * df["volume_rank"]
    ).clip(0, 100)
    df["mom_pullback_eligible"] = (
        (df["momentum_leader_score"] >= 50)
        & df["ret_1w_pct"].lt(0)
        & df["rsi0"].between(28, 68)
        & (df["volume_vs_20d"].fillna(0) >= 0.45)
    )

    # Relative strength pullback: continuous relative strength plus buyable reset.
    df["rel_strength_core_score"] = (
        0.34 * df["ret_3m_rank"]
        + 0.22 * df["ret_6m_rank"]
        + 0.18 * df["sector_strength_score"]
        + 0.14 * df["trend_alignment_score"]
        + 0.12 * df["volume_rank"]
    ).clip(0, 100)
    df["rs_pullback_depth"] = (-df["ret_1w_pct"]).clip(lower=0, upper=15)
    df["rs_pullback_entry_score"] = pct_score(df["rs_pullback_depth"], lower_is_better=False).fillna(50.0)
    df["rs_rsi_reset_score"] = (100.0 - (df["rsi0"] - 45.0).abs() * 3.0).clip(0, 100).fillna(50.0)
    df["rel_strength_pullback_score"] = (
        0.34 * df["rel_strength_core_score"]
        + 0.20 * df["rs_pullback_entry_score"]
        + 0.18 * df["sma_proximity_score"]
        + 0.14 * df["rs_rsi_reset_score"]
        + 0.14 * df["sector_strength_score"]
    ).clip(0, 100)
    df["rs_pullback_eligible"] = (
        (df["rel_strength_core_score"] >= 55)
        & df["ret_1w_pct"].lt(0)
        & (df["sector_ret_1m_median"].fillna(-99) > -8)
        & df["rsi0"].between(28, 68)
    )

    # RSI breakout / impulse: wave-3 or wave-5 expansion with volume and sector tailwind.
    df["rsi_zone_score"] = (100.0 - (df["rsi0"] - 53.0).abs() * 3.2).clip(0, 100).fillna(50.0)
    df["st_mom_score"] = pct_score(df["ret_1w_pct"], lower_is_better=False).fillna(50.0)
    df["inflect_breakout_score"] = (
        0.25 * df["rsi_zone_score"]
        + 0.22 * df["rsi_accel_pct"]
        + 0.18 * df["volume_rank"]
        + 0.15 * df["sector_strength_score"]
        + 0.12 * df["st_mom_score"]
        + 0.08 * df["breakout_location_score"]
    ).clip(0, 100)
    df["inflect_breakout_eligible"] = (
        df["rsi0"].between(38, 68)
        & df["rsi_delta_1"].gt(0)
        & df["rsi_accel"].gt(-2)
        & (df["volume_vs_20d"].fillna(0) >= 0.55)
    )

    # Elliott-wave-inspired stage classifier. This is a deterministic feature label, not a forecast.
    df["wave_accumulation_score"] = (
        0.45 * df["accumulation_location_score"]
        + 0.25 * df["rsi_acceleration_score"]
        + 0.20 * df["near_low_score"]
        + 0.10 * df["volume_rank"]
    ).clip(0, 100)
    df["wave_pullback_score"] = (
        0.42 * df["momentum_pullback_score"]
        + 0.28 * df["rel_strength_pullback_score"]
        + 0.18 * df["sma_proximity_score"]
        + 0.12 * df["rsi_cool_score"]
    ).clip(0, 100)
    df["wave_markup_score"] = (
        0.55 * df["momentum_leader_score"]
        + 0.20 * df["markup_location_score"]
        + 0.15 * df["sector_strength_score"]
        + 0.10 * df["volume_rank"]
    ).clip(0, 100)
    df["wave_breakout_score"] = (
        0.60 * df["inflect_breakout_score"]
        + 0.18 * df["breakout_location_score"]
        + 0.12 * df["ret_1m_rank"]
        + 0.10 * df["volume_rank"]
    ).clip(0, 100)
    wave_cols = ["wave_accumulation_score", "wave_pullback_score", "wave_markup_score", "wave_breakout_score"]
    wave_values = df[wave_cols].fillna(0.0).to_numpy(copy=True)
    sorted_wave = np.sort(wave_values, axis=1)
    wave_top = pd.Series(sorted_wave[:, -1], index=df.index)
    wave_second = pd.Series(sorted_wave[:, -2], index=df.index)
    df["wave_stage_margin"] = (wave_top - wave_second).clip(lower=0)
    # Reduce max-score upward bias: top stage leads, but second-best stage and
    # ambiguity matter because many stocks sit between accumulation/pullback/breakout.
    df["wave_setup_score"] = (
        0.72 * wave_top
        + 0.28 * wave_second
        - np.where(df["wave_stage_margin"] < 5.0, 4.0, 0.0)
    ).clip(0, 100).fillna(50.0)
    wave_label_map = {
        "wave_accumulation_score": "Wave 1 / accumulation",
        "wave_pullback_score": "Wave 2-4 / pullback",
        "wave_markup_score": "Wave 3 / markup leader",
        "wave_breakout_score": "Wave 3-5 / breakout",
    }
    best_wave_col = df[wave_cols].idxmax(axis=1)
    df["wave_stage"] = best_wave_col.map(wave_label_map).fillna("Neutral / transition")
    df.loc[df["wave_stage_margin"] < 3.0, "wave_stage"] = "Mixed / transition"

    df["opportunity_score"] = df[SCORE_COLUMNS].max(axis=1)

    # ── EV Master Score ──
    # High-expected-value stocks: strong signals with cross-sleeve agreement
    # and asymmetric risk/reward.  Combines signal strength, conviction,
    # payoff asymmetry, and factor alignment into a single 0-100 score.
    df["sleeve_rank_agreement"] = (
        sum(pct_score(df[col], lower_is_better=False).fillna(0) for col in SCORE_COLUMNS)
        / len(SCORE_COLUMNS)
    ).clip(0, 100)
    # Asymmetric R:R: upside room (distance to 52w high) vs downside floor (distance to 52w low)
    df["upside_potential"] = (-df["from_52w_high_pct"]).clip(lower=5, upper=100)
    df["downside_risk"] = df["from_52w_low_pct"].clip(lower=5, upper=200)
    df["rr_ratio"] = (df["upside_potential"] / df["downside_risk"]).clip(0, 10)
    df["rr_score"] = pct_score(df["rr_ratio"], lower_is_better=False).fillna(30.0)
    # Factor alignment: production factor score normalized
    if "production_factor_score" in df.columns:
        df["factor_align_score"] = pct_score(
            pd.to_numeric(df["production_factor_score"], errors="coerce"),
            lower_is_better=False,
        ).fillna(50.0)
    else:
        df["factor_align_score"] = 50.0
    df["ev_score"] = (
        0.35 * df["opportunity_score"]
        + 0.25 * df["sleeve_rank_agreement"]
        + 0.25 * df["rr_score"]
        + 0.15 * df["factor_align_score"]
    ).clip(0, 100)
    df["ev_master_eligible"] = df["ev_score"] >= 60

    df["primary_strategy"] = df[SCORE_COLUMNS].idxmax(axis=1).map(SLEEVE_LABELS)
    df["rank_in_sector"] = df.groupby("sector")["opportunity_score"].rank(method="first", ascending=False).astype(int)
    df["global_rank"] = df["opportunity_score"].rank(method="first", ascending=False).astype(int)
    df["is_top_inflection"] = (df["inflection_flag"] == 1) & (df["rsi_delta_1"] > 0) & (df["rsi_accel"] > 0)
    sorted_df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return sorted_df
