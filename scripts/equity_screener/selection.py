import pandas as pd

from .config import DIVERSIFIED_TOP_PLAN, SLEEVE_LABELS

def cap_by_sector(df: pd.DataFrame, score_col: str, limit: int, per_sector_cap: int = 3) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    counts = {}
    picks = []
    for idx, row in df.sort_values(score_col, ascending=False).iterrows():
        sector = str(row.get("sector", "Unknown"))
        if counts.get(sector, 0) >= per_sector_cap:
            continue
        counts[sector] = counts.get(sector, 0) + 1
        picks.append(idx)
        if len(picks) >= limit:
            break
    return df.loc[picks].copy()

def build_diversified_top10(df: pd.DataFrame, per_sector_cap: int = 3) -> pd.DataFrame:
    """Blend opportunity sleeves so no single strategy monopolizes the top 10."""
    if df.empty:
        return df.copy()
    sector_counts = {}
    picked = []
    picked_set = set()
    picked_source = {}

    def add_from(frame: pd.DataFrame, score_col: str, quota: int) -> None:
        nonlocal picked, picked_set, sector_counts
        added = 0
        for idx, row in frame.sort_values(score_col, ascending=False).iterrows():
            if idx in picked_set:
                continue
            sector = str(row.get("sector", "Unknown"))
            if sector_counts.get(sector, 0) >= per_sector_cap:
                continue
            picked.append(idx)
            picked_set.add(idx)
            picked_source[idx] = score_col
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            added += 1
            if added >= quota or len(picked) >= 10:
                break

    for score_col, quota in DIVERSIFIED_TOP_PLAN:
        add_from(df, score_col, quota)
        if len(picked) >= 10:
            break
    add_from(df, "opportunity_score", 10)
    out = df.loc[picked[:10]].copy()
    out["diversified_source"] = [SLEEVE_LABELS[picked_source[idx]] if picked_source[idx] in SLEEVE_LABELS else "overall opportunity" for idx in out.index]
    out["portfolio_rank"] = range(1, len(out) + 1)
    return out

def mark_diversified_top10(df: pd.DataFrame, per_sector_cap: int = 3) -> pd.DataFrame:
    """Return a copy with diversified_top10 set from the shared selector."""
    if df.empty:
        out = df.copy()
        out["diversified_top10"] = False
        return out
    out = df.copy()
    diversified = build_diversified_top10(out, per_sector_cap)
    out["diversified_top10"] = out.index.isin(diversified.index)
    return out


def final_candidate_tickers(df: pd.DataFrame) -> list[str]:
    """Return tickers that appear in final dashboard opportunity surfaces."""
    if df.empty:
        return []
    frames = [
        build_diversified_top10(df, 3),
        cap_by_sector(df, "opportunity_score", 25, 3),
        cap_by_sector(df[df["is_top_inflection"]], "rsi_value_score", 15, 3),
        cap_by_sector(df.sort_values("squeeze_laggard_score", ascending=False), "squeeze_laggard_score", 15, 3),
        cap_by_sector(df.sort_values("value_laggard_score", ascending=False), "value_laggard_score", 15, 3),
        cap_by_sector(df[df["momentum_leader_eligible"]].sort_values("momentum_leader_score", ascending=False), "momentum_leader_score", 15, 3),
        cap_by_sector(df[df["mom_pullback_eligible"]].sort_values("momentum_pullback_score", ascending=False), "momentum_pullback_score", 15, 3),
        cap_by_sector(df[df["rs_pullback_eligible"]].sort_values("rel_strength_pullback_score", ascending=False), "rel_strength_pullback_score", 25, 3),
        cap_by_sector(df[df["inflect_breakout_eligible"]].sort_values("inflect_breakout_score", ascending=False), "inflect_breakout_score", 15, 3),
        cap_by_sector(df[df["ev_master_eligible"]].sort_values("ev_score", ascending=False), "ev_score", 20, 3),
        df[df["rank_in_sector"] <= 5].sort_values(["sector", "rank_in_sector"]),
    ]
    tickers: dict[str, None] = {}
    for frame in frames:
        for ticker in frame.get("ticker", pd.Series(dtype=str)).dropna().astype(str):
            tickers[ticker.upper()] = None
    return list(tickers)
