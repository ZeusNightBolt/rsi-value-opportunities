from __future__ import annotations

import pandas as pd


def combined_top25_opportunities(df: pd.DataFrame, limit: int = 25, sector_cap: int = 3) -> pd.DataFrame:
    """Rank combined opportunities across sector, theme, and factor-basket signals.

    Returns a DataFrame with columns expected by render.py:
    combined_rank_score, opportunity_score, ev_score, production_factor_score,
    primary_keyword_factor_score, primary_keyword_factor, production_factor_basket,
    alignment_status, combined_rank_takeaway, company, ticker, etc.
    """

    def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype="float64")
        series = pd.to_numeric(df[col], errors='coerce')
        fill_val = series.median()
        if pd.isna(fill_val):
            fill_val = default
        return series.fillna(fill_val)

    df = df.copy()
    contrib_rows: list[dict] = []

    # --- Build a per-ticker composite that merges all sources ---
    # Use the existing scored dataframe fields directly, not per-basket aggregation
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue

        opp = float(row.get("opportunity_score", 0) or 0)
        ev = float(row.get("ev_score", 0) or 0)
        prod_score = float(row.get("production_factor_score", 0) or 0)
        keyword_score = float(row.get("primary_keyword_factor_score", 0) or 0)
        prod_basket = str(row.get("production_factor_basket", ""))
        keyword_factor = str(row.get("primary_keyword_factor", ""))
        company = str(row.get("company", ""))
        sector = str(row.get("sector", ""))
        rsi_val = float(row.get("rsi_value_score", 0) or 0)
        wave_val = float(row.get("wave_setup_score", 0) or 0)

        # Composite score: 40% opp + 25% EV + 15% factor + 10% theme + 10% wave adjustment
        composite = (
            0.40 * opp
            + 0.25 * ev
            + 0.15 * max(0, prod_score * 10)
            + 0.10 * max(0, keyword_score * 2)
            + 0.10 * wave_val
        )
        composite = max(0, min(100, composite))

        # Alignment: DIVERGENCE if factor basket contains avoid/broken momentum
        is_divergence = (
            "avoid" in prod_basket.lower()
            or "broken momentum" in prod_basket.lower()
        )
        alignment = "DIVERGENCE" if is_divergence else "CONFIRMATION"

        # Takeaway
        if is_divergence:
            takeaway = f"{ticker} combined score {composite:.0f} conflicts with {prod_basket}; treat as risk/reversal candidate."
        else:
            takeaway = f"{ticker} combined score {composite:.0f} confirms {prod_basket} theme alignment."

        contrib_rows.append({
            "ticker": ticker,
            "company": company,
            "combined_rank_score": round(composite, 1),
            "opportunity_score": opp,
            "ev_score": ev,
            "production_factor_score": prod_score,
            "primary_keyword_factor_score": keyword_score,
            "primary_keyword_factor": keyword_factor,
            "production_factor_basket": prod_basket,
            "alignment_status": alignment,
            "combined_rank_takeaway": takeaway,
            "sector": sector,
            "display_close": row.get("display_close"),
            "price_source": row.get("price_source"),
            "rsi0": row.get("rsi0"),
            "four_h_timestamp": row.get("four_h_timestamp"),
            "latest_daily_timestamp": row.get("latest_daily_timestamp"),
            "market_cap": row.get("market_cap"),
            "short_pct_float": row.get("short_pct_float"),
            "from_52w_low_pct": row.get("from_52w_low_pct"),
            "peer_lag_1m_pct": row.get("peer_lag_1m_pct"),
            "wave_stage": row.get("wave_stage"),
        })

    if not contrib_rows:
        return pd.DataFrame()

    combined_df = pd.DataFrame(contrib_rows)

    # --- Apply sector cap (use built-in head per group) ---
    combined_df = (
        combined_df
        .sort_values(["alignment_status", "combined_rank_score"], ascending=[True, False])
        .groupby("sector")
        .head(sector_cap)
    )

    # --- Apply global limit ---
    combined_df = combined_df.head(limit).copy()
    combined_df["display_rank"] = range(1, len(combined_df) + 1)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df
