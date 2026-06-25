from __future__ import annotations

import pandas as pd

from .divergence import top10_factor_alignment


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    series = df[col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.Series(pd.to_numeric(series, errors="coerce"), index=df.index).fillna(default).astype("float64")


def combined_top25_opportunities(df: pd.DataFrame, limit: int = 25, sector_cap: int = 4) -> pd.DataFrame:
    """Rank combined opportunities across sector, theme, and factor-basket signals.

    The rank is intentionally confirmation-aware: names with strong opportunity,
    EV, production-factor, and keyword-theme scores move up, while names in
    Broken Momentum / Avoid remain visible as divergence candidates but are
    penalized versus clean confirmations.
    """
    if df.empty:
        return df.copy()

    out = top10_factor_alignment(df).copy()
    opp = _num(out, "opportunity_score")
    ev = _num(out, "ev_score") if "ev_score" in out.columns else opp
    factor = _num(out, "production_factor_score", 50.0)
    theme = _num(out, "primary_keyword_factor_score", 50.0)
    alignment_bonus = out["alignment_status"].map({"CONFIRMATION": 8.0, "DIVERGENCE": -12.0}).fillna(0.0)

    out["combined_rank_score"] = (
        0.45 * opp
        + 0.25 * ev
        + 0.15 * factor
        + 0.15 * theme
        + alignment_bonus
    ).clip(0, 100)
    out["combined_rank_takeaway"] = out.apply(
        lambda r: (
            f"{r.get('alignment_status', '')}: Opp {float(r.get('opportunity_score', 0) or 0):.0f}, "
            f"EV {float(r.get('ev_score', r.get('opportunity_score', 0)) or 0):.0f}, "
            f"factor {float(r.get('production_factor_score', 0) or 0):.0f}, "
            f"theme {float(r.get('primary_keyword_factor_score', 0) or 0):.0f}."
        ),
        axis=1,
    )

    out = out.sort_values(["combined_rank_score", "opportunity_score"], ascending=[False, False])
    if sector_cap > 0 and "sector" in out.columns:
        out = out.groupby("sector", group_keys=False, dropna=False).head(sector_cap)
    return out.head(limit).reset_index(drop=True)
