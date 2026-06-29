import math

import pandas as pd

def clean_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value) or math.isinf(float(value)):
            return None
        return float(value)
    except Exception:
        return None

def record(row) -> dict:
    keys = [
        "global_rank", "rank_in_sector", "portfolio_rank", "display_rank", "sector", "ticker", "company", "market_cap", "four_h_close", "display_close", "price_source", "latest_daily_close",
        "latest_polygon_price", "latest_polygon_price_source", "latest_polygon_price_timestamp", "latest_polygon_price_status", "warehouse_display_close",
        "diversified_source",
        "raw_opportunity_score", "quality_penalty", "thin_volume_penalty", "broken_trend_penalty", "crash_penalty",
        "opportunity_score", "rsi_value_score", "squeeze_laggard_score", "value_laggard_score", "momentum_leader_score", "momentum_pullback_score", "rel_strength_pullback_score", "inflect_breakout_score", "wave_setup_score", "ev_score",
        "rsi_acceleration_score", "composite_value_score", "rel_strength_core_score", "price_position_52w", "wave_stage", "wave_stage_margin", "wave_accumulation_score", "wave_pullback_score", "wave_markup_score", "wave_breakout_score", "rsi0", "rsi1", "rsi2", "rsi3", "rsi4", "rsi5", "rsi_delta_1",
        "prior_delta_3_avg", "rsi_accel", "inflection_flag", "yf_forward_pe", "yf_trailing_pe",
        "yf_price_to_book", "yf_peg_ratio", "from_52w_high_pct", "from_52w_low_pct", "short_pct_float",
        "ret_1w_pct", "ret_1m_pct", "ret_3m_pct", "ret_6m_pct", "ret_ytd_pct",
        "sector_ret_1m_median", "peer_lag_1m_pct", "peer_lag_3m_pct",
        "near_low_score", "short_score", "peer_lag_score", "sentiment_score", "value_grade",
        "growth_grade", "momentum_grade", "primary_strategy", "production_factor_basket", "production_factor_score",
        "production_theme", "primary_keyword_factor", "primary_keyword_factor_score", "keyword_factor_baskets", "four_h_timestamp", "latest_daily_timestamp",
    ]
    out = {}
    for key in keys:
        value = row.get(key)
        if key in {"sector", "ticker", "company", "price_source", "latest_polygon_price_source", "latest_polygon_price_status", "value_grade", "growth_grade", "momentum_grade", "primary_strategy", "diversified_source", "wave_stage", "production_factor_basket", "production_theme", "primary_keyword_factor", "keyword_factor_baskets"}:
            out[key] = None if pd.isna(value) else str(value)
        elif key in {"four_h_timestamp", "latest_daily_timestamp"}:
            out[key] = str(value)
        elif key in {"global_rank", "rank_in_sector", "portfolio_rank", "display_rank", "inflection_flag"}:
            out[key] = None if pd.isna(value) else int(value)
        else:
            out[key] = clean_float(value)
    return out
