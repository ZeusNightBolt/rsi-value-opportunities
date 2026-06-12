#!/usr/bin/env python3
import argparse
import datetime as dt
import html
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path.home() / "market-data" / "market_data.duckdb"
ENV_PATH = Path.home() / ".hermes" / ".env"
DOCS_DIR = PROJECT_DIR / "docs"
DATA_DIR = PROJECT_DIR / "data"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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


def query_candidates(price_filter: float) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = r"""
with rsi_hist as (
  select ticker, timestamp, rsi_14, close,
         row_number() over(partition by ticker order by timestamp desc) rn
  from technical_indicators
  where timeframe='4h' and rsi_14 is not null and close is not null
), piv as (
  select ticker,
    max(case when rn=1 then timestamp end) ts0,
    max(case when rn=1 then close end) close0,
    max(case when rn=1 then rsi_14 end) rsi0,
    max(case when rn=2 then rsi_14 end) rsi1,
    max(case when rn=3 then rsi_14 end) rsi2,
    max(case when rn=4 then rsi_14 end) rsi3,
    max(case when rn=5 then rsi_14 end) rsi4,
    max(case when rn=6 then rsi_14 end) rsi5,
    count(*) n
  from rsi_hist where rn<=6 group by ticker
), latest_daily_ts as (
  select max(timestamp) as max_ts from daily_bars
), daily_52w as (
  select d.ticker, min(d.low) low_52w, max(d.high) high_52w
  from daily_bars d, latest_daily_ts l
  where d.timestamp >= l.max_ts - 31536000000
    and d.low is not null and d.high is not null
  group by d.ticker
), base as (
  select distinct
    s.sector,
    s.ticker,
    coalesce(s.company_name, s.holding_name) company,
    s.market_cap,
    s.exchange,
    s.industry,
    s.sic_description,
    s.sentiment_score,
    s.short_pct_float,
    coalesce(s.from_52w_high_pct, case when d.high_52w > 0 then ((p.close0 / d.high_52w) - 1.0) * 100.0 end) from_52w_high_pct,
    coalesce(s.from_52w_low_pct, case when d.low_52w > 0 then ((p.close0 / d.low_52w) - 1.0) * 100.0 end) from_52w_low_pct,
    s.price_vs_sma20_pct,
    s.price_vs_sma50_pct,
    s.price_vs_sma200_pct,
    s.volume_vs_20d,
    e.yf_forward_pe,
    e.yf_trailing_pe,
    e.yf_price_to_book,
    e.yf_peg_ratio,
    e.yf_dividend_yield,
    e.value_grade,
    e.growth_grade,
    e.momentum_grade,
    f.dolt_value_score,
    f.production_factor_score,
    f.quant_factor_score,
    f.ret_1w_pct,
    f.ret_1m_pct,
    f.ret_3m_pct,
    f.ret_6m_pct,
    f.ret_ytd_pct,
    p.ts0,
    to_timestamp(p.ts0/1000) four_h_timestamp,
    p.close0 four_h_close,
    p.rsi0, p.rsi1, p.rsi2, p.rsi3, p.rsi4, p.rsi5,
    (p.rsi0-p.rsi1) rsi_delta_1,
    ((p.rsi1-p.rsi2)+(p.rsi2-p.rsi3)+(p.rsi3-p.rsi4))/3.0 prior_delta_3_avg,
    ((p.rsi0-p.rsi1) - (((p.rsi1-p.rsi2)+(p.rsi2-p.rsi3)+(p.rsi3-p.rsi4))/3.0)) rsi_accel,
    case when (p.rsi0-p.rsi1)>0 and (((p.rsi1-p.rsi2)+(p.rsi2-p.rsi3)+(p.rsi3-p.rsi4))/3.0)<0 then 1 else 0 end inflection_flag
  from v_vti_sector_universe_5b s
  join piv p on s.ticker=p.ticker
  left join vti_daily_enriched_latest e on s.ticker=e.ticker
  left join v_vti_factor_production_scores_5b f on s.ticker=f.ticker
  left join daily_52w d on s.ticker=d.ticker
  where s.market_cap >= 5000000000
    and p.close0 < ?
    and s.sector is not null and s.sector <> ''
    and p.n >= 5
)
select * from base
"""
    df = con.execute(q, [price_filter]).fetchdf()
    con.close()
    return df


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
    """Blend multiple opportunity sleeves; do not let RSI monopolize the top 10."""
    if df.empty:
        return df.copy()
    sector_counts = {}
    picked = []
    picked_set = set()

    def add_from(frame: pd.DataFrame, score_col: str, quota: int) -> None:
        nonlocal picked, picked_set, sector_counts
        for idx, row in frame.sort_values(score_col, ascending=False).iterrows():
            if idx in picked_set:
                continue
            sector = str(row.get("sector", "Unknown"))
            if sector_counts.get(sector, 0) >= per_sector_cap:
                continue
            picked.append(idx)
            picked_set.add(idx)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len([i for i in picked if i in frame.index]) >= quota or len(picked) >= 10:
                break

    # 4 best all-around, 3 short/lows/peer-lag, 3 value-laggards. Fill any remaining by overall score.
    add_from(df, "opportunity_score", 4)
    add_from(df, "squeeze_laggard_score", 7)
    add_from(df, "value_laggard_score", 10)
    add_from(df, "opportunity_score", 10)
    out = df.loc[picked[:10]].copy()
    out["portfolio_rank"] = range(1, len(out) + 1)
    return out


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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

    df["rsi_accel_pct"] = pct_score(df["rsi_accel"], lower_is_better=False).fillna(50.0)
    df["rsi_delta_pct"] = pct_score(df["rsi_delta_1"], lower_is_better=False).fillna(50.0)
    df["grind_bonus"] = np.where(
        (df["inflection_flag"] == 1) & (df["rsi_delta_1"] >= 5) & (df["rsi_accel"] >= 10),
        15.0,
        0.0,
    )
    df["rsi_acceleration_score"] = (0.70 * df["rsi_accel_pct"] + 0.20 * df["rsi_delta_pct"] + df["grind_bonus"]).clip(0, 100)
    df["rsi_value_score"] = 0.65 * df["rsi_acceleration_score"] + 0.35 * df["composite_value_score"]

    df["sector_ret_1m_median"] = df.groupby("sector")["ret_1m_pct"].transform("median")
    df["sector_ret_3m_median"] = df.groupby("sector")["ret_3m_pct"].transform("median")
    df["peer_lag_1m_pct"] = df["sector_ret_1m_median"] - df["ret_1m_pct"]
    df["peer_lag_3m_pct"] = df["sector_ret_3m_median"] - df["ret_3m_pct"]
    df["peer_lag_score"] = (0.65 * pct_score(df["peer_lag_1m_pct"], lower_is_better=False).fillna(50.0) + 0.35 * pct_score(df["peer_lag_3m_pct"], lower_is_better=False).fillna(50.0)).clip(0, 100)
    df["peer_rally_laggard_flag"] = ((df["sector_ret_1m_median"] > 3.0) & (df["peer_lag_1m_pct"] > 5.0)).astype(int)

    df["short_score"] = pct_score(df["short_pct_float"], lower_is_better=False).fillna(35.0)
    df["near_low_score"] = pct_score(df["from_52w_low_pct"], lower_is_better=True).fillna(50.0)
    df["oversold_not_broken_score"] = np.where(df["rsi0"].between(25, 55), 70.0, 40.0)
    df["squeeze_laggard_score"] = (
        0.35 * df["short_score"]
        + 0.30 * df["near_low_score"]
        + 0.25 * df["peer_lag_score"]
        + 0.10 * df["oversold_not_broken_score"]
        + np.where((df["peer_rally_laggard_flag"] == 1) & (df["short_pct_float"] >= 5), 10.0, 0.0)
    ).clip(0, 100)
    df["value_laggard_score"] = (0.45 * df["composite_value_score"] + 0.35 * df["peer_lag_score"] + 0.20 * df["near_low_score"]).clip(0, 100)

    score_cols = ["rsi_value_score", "squeeze_laggard_score", "value_laggard_score"]
    df["opportunity_score"] = df[score_cols].max(axis=1)
    labels = {
        "rsi_value_score": "RSI inflection + value",
        "squeeze_laggard_score": "shorted near lows / peer lag",
        "value_laggard_score": "cheap peer laggard",
    }
    df["primary_strategy"] = df[score_cols].idxmax(axis=1).map(labels)
    df["rank_in_sector"] = df.groupby("sector")["opportunity_score"].rank(method="first", ascending=False).astype(int)
    df["global_rank"] = df["opportunity_score"].rank(method="first", ascending=False).astype(int)
    df["is_top_inflection"] = (df["inflection_flag"] == 1) & (df["rsi_delta_1"] > 0) & (df["rsi_accel"] > 0)
    diversified = cap_by_sector(df, "opportunity_score", 10, 3)
    df["diversified_top10"] = df.index.isin(diversified.index)
    return df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)

def call_llm(prompt: str) -> str:
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if deepseek_key:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
        model = "deepseek-chat"
    elif openrouter_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ZeusNightBolt/rsi-value-opportunities",
            "X-Title": "RSI Value Opportunities Dashboard",
        }
        model = "deepseek/deepseek-chat-v3-0324"
    else:
        return "LLM unavailable: DEEPSEEK_API_KEY and OPENROUTER_API_KEY were not found."

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a hedge-fund research assistant. Be concise, skeptical, and compliance-safe. Do not give investment advice. Analyze the long setup using only supplied data; clearly label unknowns.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return f"LLM call failed: {type(exc).__name__}: {exc}"


def analyze_top_inflections(df: pd.DataFrame, top_n: int, force_refresh: bool) -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "llm_analysis.json"
    cache = {}
    if cache_path.exists() and not force_refresh:
        try:
            cache = {item["ticker"]: item for item in json.loads(cache_path.read_text())}
        except Exception:
            cache = {}

    candidates = build_diversified_top10(df, 3).head(top_n)
    analyses = []
    for _, row in candidates.iterrows():
        ticker = str(row["ticker"])
        cache_key = f"{ticker}:{row['ts0']}:{round(float(row['opportunity_score']), 2)}:{row.get('primary_strategy')}"
        cached = cache.get(ticker)
        if cached and cached.get("cache_key") == cache_key:
            analyses.append(cached)
            continue
        payload = {
            "ticker": ticker,
            "company": row.get("company"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "primary_strategy": row.get("primary_strategy"),
            "price": round(float(row.get("four_h_close")), 4),
            "market_cap_bn": round(float(row.get("market_cap")) / 1e9, 2),
            "opportunity_score": round(float(row.get("opportunity_score")), 1),
            "rsi_value_score": round(float(row.get("rsi_value_score")), 1),
            "squeeze_laggard_score": round(float(row.get("squeeze_laggard_score")), 1),
            "value_laggard_score": round(float(row.get("value_laggard_score")), 1),
            "composite_value_score": round(float(row.get("composite_value_score")), 1),
            "rsi_acceleration_score": round(float(row.get("rsi_acceleration_score")), 1),
            "rsi_current": round(float(row.get("rsi0")), 1),
            "rsi_delta_1": round(float(row.get("rsi_delta_1")), 1),
            "prior_rsi_delta_3_bar_avg": round(float(row.get("prior_delta_3_avg")), 1),
            "rsi_acceleration": round(float(row.get("rsi_accel")), 1),
            "short_pct_float": None if pd.isna(row.get("short_pct_float")) else round(float(row.get("short_pct_float")), 1),
            "from_52w_low_pct": None if pd.isna(row.get("from_52w_low_pct")) else round(float(row.get("from_52w_low_pct")), 1),
            "from_52w_high_pct": None if pd.isna(row.get("from_52w_high_pct")) else round(float(row.get("from_52w_high_pct")), 1),
            "ret_1m_pct": None if pd.isna(row.get("ret_1m_pct")) else round(float(row.get("ret_1m_pct")), 1),
            "sector_ret_1m_median": None if pd.isna(row.get("sector_ret_1m_median")) else round(float(row.get("sector_ret_1m_median")), 1),
            "peer_lag_1m_pct": None if pd.isna(row.get("peer_lag_1m_pct")) else round(float(row.get("peer_lag_1m_pct")), 1),
            "forward_pe": None if pd.isna(row.get("yf_forward_pe")) else round(float(row.get("yf_forward_pe")), 2),
            "trailing_pe": None if pd.isna(row.get("yf_trailing_pe")) else round(float(row.get("yf_trailing_pe")), 2),
            "price_to_book": None if pd.isna(row.get("yf_price_to_book")) else round(float(row.get("yf_price_to_book")), 2),
            "peg": None if pd.isna(row.get("yf_peg_ratio")) else round(float(row.get("yf_peg_ratio")), 2),
            "sentiment_score": None if pd.isna(row.get("sentiment_score")) else round(float(row.get("sentiment_score")), 2),
        }
        prompt = (
            "Analyze this candidate as a possible LONG setup for a research dashboard. "
            "Do not force an RSI-only story. Use the named primary strategy and supplied data: RSI inflection if present, but also short interest, proximity to 52-week lows, valuation, peer/sector lag, and whether peers have recently rallied. "
            "Return exactly 5 bullets with labels: Setup, Why it can work, What can break it, Confirming evidence to watch, Bottom line. "
            "Be specific and skeptical. No trade recommendation, no target price. Supplied data:\n"
            + json.dumps(payload, indent=2)
        )
        text = call_llm(prompt)
        item = {
            "ticker": ticker,
            "cache_key": cache_key,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input": payload,
            "analysis": text,
        }
        analyses.append(item)
        time.sleep(0.5)
    cache_path.write_text(json.dumps(analyses, indent=2, default=str))
    return analyses

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
        "global_rank", "rank_in_sector", "sector", "ticker", "company", "market_cap", "four_h_close",
        "opportunity_score", "rsi_value_score", "squeeze_laggard_score", "value_laggard_score",
        "rsi_acceleration_score", "composite_value_score", "rsi0", "rsi_delta_1",
        "prior_delta_3_avg", "rsi_accel", "inflection_flag", "yf_forward_pe", "yf_trailing_pe",
        "yf_price_to_book", "yf_peg_ratio", "from_52w_high_pct", "from_52w_low_pct", "short_pct_float",
        "ret_1m_pct", "ret_3m_pct", "sector_ret_1m_median", "peer_lag_1m_pct", "peer_lag_3m_pct",
        "near_low_score", "short_score", "peer_lag_score", "sentiment_score", "value_grade",
        "growth_grade", "momentum_grade", "primary_strategy", "four_h_timestamp",
    ]
    out = {}
    for key in keys:
        value = row.get(key)
        if key in {"sector", "ticker", "company", "value_grade", "growth_grade", "momentum_grade", "primary_strategy"}:
            out[key] = None if pd.isna(value) else str(value)
        elif key == "four_h_timestamp":
            out[key] = str(value)
        elif key in {"global_rank", "rank_in_sector", "inflection_flag"}:
            out[key] = None if pd.isna(value) else int(value)
        else:
            out[key] = clean_float(value)
    return out


def fmt_num(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_money(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"${float(value):.2f}"


def fmt_bn(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"${float(value)/1e9:.1f}B"


def render_bar(value, cls=""):
    v = 0 if value is None or (isinstance(value, float) and math.isnan(value)) else max(0, min(100, float(value)))
    return f'<span class="bar {cls}"><i style="width:{v:.1f}%"></i></span><b>{v:.1f}</b>'


def finviz_url(ticker: str) -> str:
    return "https://finviz.com/stock?t=" + urllib.parse.quote(str(ticker).strip().upper(), safe=".-")


def ticker_link(ticker: str, cls: str = "ticker-link") -> str:
    safe_ticker = html.escape(str(ticker).strip().upper())
    safe_href = html.escape(finviz_url(ticker), quote=True)
    return f'<a class="{cls}" href="{safe_href}" target="_blank" rel="noopener noreferrer">{safe_ticker}</a>'


def render_dashboard(df: pd.DataFrame, analyses: list[dict], price_filter: float) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now_et = dt.datetime.now().astimezone()
    latest_ts = str(df["four_h_timestamp"].max()) if not df.empty else "unknown"
    diversified_top = build_diversified_top10(df, 3)
    top = cap_by_sector(df, "opportunity_score", 25, 3)
    top_sector = df[df["rank_in_sector"] <= 5].sort_values(["sector", "rank_in_sector"])
    inflect = cap_by_sector(df[df["is_top_inflection"]], "rsi_value_score", 15, 3)
    squeeze = cap_by_sector(df.sort_values("squeeze_laggard_score", ascending=False), "squeeze_laggard_score", 15, 3)
    laggards = cap_by_sector(df.sort_values("value_laggard_score", ascending=False), "value_laggard_score", 15, 3)

    payload = {
        "generated_at": now_et.isoformat(),
        "latest_4h_timestamp": latest_ts,
        "price_filter": price_filter,
        "universe_count": int(len(df)),
        "sector_count": int(df["sector"].nunique()) if not df.empty else 0,
        "top_diversified": [record(r) for _, r in diversified_top.iterrows()],
        "top": [record(r) for _, r in top.iterrows()],
        "inflections": [record(r) for _, r in inflect.iterrows()],
        "squeeze_laggards": [record(r) for _, r in squeeze.iterrows()],
        "value_laggards": [record(r) for _, r in laggards.iterrows()],
        "by_sector": [record(r) for _, r in top_sector.iterrows()],
        "llm_analysis": analyses,
    }
    (DATA_DIR / "dashboard_data.json").write_text(json.dumps(payload, indent=2, default=str))
    (DOCS_DIR / "dashboard_data.json").write_text(json.dumps(payload, indent=2, default=str))

    def row_html(r, rank_field="global_rank"):
        return (
            "<tr>"
            f"<td>{int(r[rank_field]) if rank_field in r and not pd.isna(r[rank_field]) else ''}</td>"
            f"<td><strong>{ticker_link(r['ticker'])}</strong><small>{html.escape(str(r['company'])[:42])}</small></td>"
            f"<td>{html.escape(str(r['sector']))}<small>{html.escape(str(r.get('primary_strategy', '')))}</small></td>"
            f"<td>{fmt_money(r['four_h_close'])}</td>"
            f"<td>{fmt_bn(r['market_cap'])}</td>"
            f"<td>{render_bar(r['opportunity_score'])}</td>"
            f"<td>{render_bar(r.get('rsi_value_score'), 'hot')}</td>"
            f"<td>{render_bar(r.get('squeeze_laggard_score'), 'short')}</td>"
            f"<td>{render_bar(r.get('value_laggard_score'), 'value')}</td>"
            f"<td>{fmt_num(r.get('short_pct_float'))}%</td>"
            f"<td>{fmt_num(r.get('from_52w_low_pct'))}%</td>"
            f"<td>{fmt_num(r.get('peer_lag_1m_pct'))}%</td>"
            f"<td>{fmt_num(r.get('rsi_accel'))}</td>"
            "</tr>"
        )

    top_rows = [row_html(r) for _, r in top.iterrows()]
    div_rows = []
    for display_rank, (_, r) in enumerate(diversified_top.iterrows(), 1):
        rr = r.copy()
        rr["display_rank"] = display_rank
        div_rows.append(row_html(rr, "display_rank"))
    inflect_rows = [row_html(r) for _, r in inflect.iterrows()]
    squeeze_rows = [row_html(r) for _, r in squeeze.iterrows()]
    laggard_rows = [row_html(r) for _, r in laggards.iterrows()]

    analysis_cards = []
    for item in analyses:
        inp = item["input"]
        text = html.escape(item["analysis"]).replace("\n", "<br>")
        analysis_cards.append(
            f"<article class='card'><div class='card-head'><span>{ticker_link(item['ticker'])}</span>"
            f"<em>{html.escape(str(inp.get('sector')))} · ${inp.get('price')}</em></div>"
            f"<div class='metrics'>{html.escape(str(inp.get('primary_strategy')))} · Score {inp.get('opportunity_score')} · short {inp.get('short_pct_float')}% · peer lag {inp.get('peer_lag_1m_pct')}%</div>"
            f"<p>{text}</p></article>"
        )

    sector_sections = []
    for sector, sdf in top_sector.groupby("sector"):
        items = []
        for _, r in sdf.iterrows():
            items.append(
                f"<li><b>{ticker_link(r['ticker'])}</b> {fmt_money(r['four_h_close'])} "
                f"score {fmt_num(r['opportunity_score'])} · {html.escape(str(r.get('primary_strategy')))} · short {fmt_num(r.get('short_pct_float'))}% · lag {fmt_num(r.get('peer_lag_1m_pct'))}%</li>"
            )
        sector_sections.append(f"<section class='sector'><h3>{html.escape(str(sector))}</h3><ol>{''.join(items)}</ol></section>")

    header = "<table><thead><tr><th>#</th><th>Ticker</th><th>Sector / Sleeve</th><th>4h Px</th><th>MCap</th><th>Opp</th><th>RSI</th><th>Short/Lows</th><th>Value/Lag</th><th>Short</th><th>From Low</th><th>Peer Lag 1M</th><th>RSI Accel</th></tr></thead><tbody>"
    css = """
:root{--bg:#080808;--panel:#101010;--panel2:#151515;--text:#e8e4d8;--muted:#8a867b;--line:#2a2822;--amber:#e6b422;--green:#40c463;--red:#ff5c5c;--purple:#b58cff;--cyan:#47d7ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:'JetBrains Mono','SFMono-Regular',Consolas,monospace;font-size:13px}header{border-bottom:1px solid var(--line);padding:18px 22px;background:#0d0d0d;position:sticky;top:0;z-index:2}h1{font-size:18px;margin:0 0 8px;color:var(--amber);letter-spacing:.04em}h2{font-size:15px;margin:28px 0 12px;color:var(--amber);text-transform:uppercase}h3{font-size:13px;color:var(--amber)}.status{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted)}.dot{color:var(--green)}main{padding:18px 22px;max-width:1600px;margin:0 auto}.note{border:1px solid var(--line);padding:12px;background:var(--panel);color:var(--muted);line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}.card,.sector{border:1px solid var(--line);background:var(--panel);padding:12px}.card-head{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:8px}.card-head span{color:var(--amber);font-size:16px;font-weight:bold}.card-head em{font-style:normal;color:var(--muted)}.metrics{color:var(--green);margin-bottom:8px}.card p{line-height:1.55;margin:0;color:#d8d2c3}table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{border:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{color:var(--amber);font-weight:600;background:#0e0e0e;position:sticky;top:76px}td small{display:block;color:var(--muted);font-size:11px;margin-top:3px}a.ticker-link{color:var(--amber);text-decoration:none;border-bottom:1px solid rgba(230,180,34,.45)}a.ticker-link:hover{color:var(--green);border-bottom-color:var(--green)}.bar{display:inline-block;width:74px;height:7px;background:#242018;margin-right:8px;vertical-align:middle}.bar i{display:block;height:100%;background:var(--amber)}.bar.hot i{background:var(--green)}.bar.value i{background:var(--purple)}.bar.short i{background:var(--cyan)}.sector ol{margin:0;padding-left:22px}.sector li{margin:6px 0;line-height:1.45}.footer{color:var(--muted);font-size:11px;margin:28px 0}.pill{border:1px solid var(--line);padding:3px 6px;color:var(--amber);display:inline-block;margin-right:6px}a{color:var(--amber)}
"""
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSI Value Opportunities</title><style>{css}</style></head>
<body><header><h1>MULTI-SLEEVE VALUE OPPORTUNITIES</h1><div class="status"><span class="dot">● LIVE</span><span>generated {html.escape(now_et.strftime('%Y-%m-%d %H:%M %Z'))}</span><span>latest 4h {html.escape(latest_ts)}</span><span>price &lt; ${price_filter:.0f}</span><span>{len(df)} names / {df['sector'].nunique() if not df.empty else 0} sectors</span><span>top 10 capped at max 3/sector</span></div></header>
<main>
<div class="note"><span class="pill">Method</span> Multi-sleeve rank: RSI inflection + value, shorted-near-lows / peer lag, and cheap peer laggards. The top 10 blends three sleeves and is diversified with a hard cap of 3 stocks per sector. Known numeric data comes from the local Polygon/DuckDB warehouse; LLM notes are qualitative research commentary, not investment advice.</div>
<h2>Multi-sleeve top 10 opportunities</h2>{header}{''.join(div_rows)}</tbody></table>
<h2>DeepSeek review: diversified top 10</h2><div class="grid">{''.join(analysis_cards)}</div>
<h2>RSI inflection sleeve</h2>{header}{''.join(inflect_rows)}</tbody></table>
<h2>Shorted near lows / peer lag sleeve</h2>{header}{''.join(squeeze_rows)}</tbody></table>
<h2>Cheap peer laggard sleeve</h2>{header}{''.join(laggard_rows)}</tbody></table>
<h2>Top 25 diversified ranked opportunities</h2>{header}{''.join(top_rows)}</tbody></table>
<h2>Top by sector</h2><div class="grid">{''.join(sector_sections)}</div>
<div class="footer">Known: numeric data from local Polygon/DuckDB warehouse. Estimated: composite scores from normalized warehouse fields. Unknown: forward catalysts beyond supplied warehouse fields unless LLM explicitly labels them unknown.</div>
</main></body></html>"""
    (DOCS_DIR / "index.html").write_text(content)

def git_commit_push() -> None:
    subprocess.run(["git", "add", "README.md", ".gitignore", "run_daily.sh", "scripts/build_dashboard.py", "docs/index.html", "docs/dashboard_data.json", "data/dashboard_data.json", "data/llm_analysis.json", "data/scored_candidates.csv"], cwd=PROJECT_DIR, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True, capture_output=True, check=True).stdout.strip()
    if not status:
        print("git: no changes to commit")
        return
    subprocess.run(["git", "commit", "-m", "Update RSI value opportunities dashboard"], cwd=PROJECT_DIR, check=True)
    remotes = subprocess.run(["git", "remote"], cwd=PROJECT_DIR, text=True, capture_output=True, check=True).stdout.strip().splitlines()
    if "origin" in remotes:
        subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_DIR, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-filter", type=float, default=30.0)
    parser.add_argument("--top-llm", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--force-llm", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    load_env_file(ENV_PATH)
    df = query_candidates(args.price_filter)
    df = score_candidates(df)
    analyses = [] if args.no_llm else analyze_top_inflections(df, args.top_llm, args.force_llm)
    render_dashboard(df, analyses, args.price_filter)
    csv_path = DATA_DIR / "scored_candidates.csv"
    df.to_csv(csv_path, index=False)
    if args.push:
        git_commit_push()
    print(json.dumps({
        "ok": True,
        "price_filter": args.price_filter,
        "universe_count": int(len(df)),
        "sector_count": int(df["sector"].nunique()) if not df.empty else 0,
        "inflection_count": int(df["is_top_inflection"].sum()) if not df.empty else 0,
        "llm_analyses": len(analyses),
        "dashboard": str(DOCS_DIR / "index.html"),
        "data": str(DATA_DIR / "dashboard_data.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
