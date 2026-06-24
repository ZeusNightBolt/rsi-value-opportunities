#!/usr/bin/env python3
import argparse
import datetime as dt
import html
import json
import math
import os
import re
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
), latest_daily as (
  select ticker, timestamp as daily_ts, close as daily_close
  from (
    select ticker, timestamp, close,
           row_number() over(partition by ticker order by timestamp desc) rn
    from daily_bars
    where close is not null
  ) where rn = 1
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
    coalesce(s.from_52w_high_pct, case when d.high_52w > 0 then ((case when ld.daily_ts > p.ts0 then ld.daily_close else p.close0 end / d.high_52w) - 1.0) * 100.0 end) from_52w_high_pct,
    coalesce(s.from_52w_low_pct, case when d.low_52w > 0 then ((case when ld.daily_ts > p.ts0 then ld.daily_close else p.close0 end / d.low_52w) - 1.0) * 100.0 end) from_52w_low_pct,
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
    f.production_factor_basket,
    f.production_theme,
    f.primary_keyword_factor,
    f.primary_keyword_factor_score,
    f.keyword_factor_baskets,
    f.quant_factor_score,
    f.ret_1w_pct,
    f.ret_1m_pct,
    f.ret_3m_pct,
    f.ret_6m_pct,
    f.ret_ytd_pct,
    p.ts0,
    to_timestamp(p.ts0/1000) four_h_timestamp,
    p.close0 four_h_close,
    ld.daily_ts latest_daily_ts,
    to_timestamp(ld.daily_ts/1000) latest_daily_timestamp,
    ld.daily_close latest_daily_close,
    case when ld.daily_ts > p.ts0 then ld.daily_close else p.close0 end display_close,
    case when ld.daily_ts > p.ts0 then 'daily_close_newer_than_4h' else '4h_close' end price_source,
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
  left join latest_daily ld on s.ticker=ld.ticker
  where s.market_cap >= 5000000000
    and case when ld.daily_ts > p.ts0 then ld.daily_close else p.close0 end < ?
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

    # 4 best all-around, 3 short/lows/peer-lag, 3 value-laggards, 3 momentum pullbacks.
    # Fill any remaining by overall score.
    add_from(df, "opportunity_score", 4)
    add_from(df, "squeeze_laggard_score", 7)
    add_from(df, "value_laggard_score", 10)
    add_from(df, "momentum_pullback_score", 10)
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

    # ── Momentum Pullback Score ──
    # Strong 6-month upward momentum + 1-2 week pullback + consolidating near
    # moving averages.  This screen finds stocks like NBIS: strong multi-month
    # uptrend that just sold off short-term and is coiling into SMAs — a
    # continuation setup, not a reversal.
    for c in ["ret_6m_pct", "ret_1w_pct", "price_vs_sma50_pct", "price_vs_sma200_pct"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df["ret_6m_pct_score"] = pct_score(df["ret_6m_pct"], lower_is_better=False).fillna(50.0)
    # Pullback depth: scored where 5-40% below 52w high (meaningful dip, not broken)
    df["pullback_depth"] = (-df["from_52w_high_pct"]).clip(lower=0)
    df["pullback_score"] = pct_score(
        df["pullback_depth"].where(df["pullback_depth"].between(5, 40)),
        lower_is_better=False,
    ).fillna(0.0)
    # Consolidation: proximity to SMA50 + SMA200 (closer → tighter coil)
    df["sma_proximity_score"] = (
        0.6 * pct_score(df["price_vs_sma50_pct"].abs(), lower_is_better=True).fillna(50.0)
        + 0.4 * pct_score(df["price_vs_sma200_pct"].abs(), lower_is_better=True).fillna(50.0)
    ).clip(0, 100)
    # RSI cooling tent: peaks at ~45-50, decays toward 30 and 70
    df["rsi_cool_score"] = np.where(
        df["rsi0"].between(30, 65),
        (100.0 - abs(df["rsi0"] - 47.5) * 5.0).clip(0, 100),
        0.0,
    )
    # Volume contraction during pullback (sub-1.0 = drying up)
    df["vol_contract_score"] = pct_score(
        df["volume_vs_20d"].clip(upper=1.5), lower_is_better=True,
    ).fillna(50.0)
    # Gate: must have momentum (6m >10%), recent pullback (1w <0%), off highs
    df["mom_pullback_eligible"] = (
        df["ret_6m_pct"].gt(10)
        & df["ret_1w_pct"].lt(0)
        & df["from_52w_high_pct"].lt(-5)
        & df["rsi0"].between(30, 65)
    )
    df["momentum_pullback_score"] = np.where(
        df["mom_pullback_eligible"],
        (
            0.30 * df["ret_6m_pct_score"]
            + 0.25 * df["pullback_score"]
            + 0.20 * df["sma_proximity_score"]
            + 0.15 * df["rsi_cool_score"]
            + 0.10 * df["vol_contract_score"]
        ).clip(0, 100),
        0.0,
    )

    score_cols = ["rsi_value_score", "squeeze_laggard_score", "value_laggard_score", "momentum_pullback_score"]
    df["opportunity_score"] = df[score_cols].max(axis=1)

    # ── EV Master Score ──
    # High-expected-value stocks: strong signals with cross-sleeve agreement
    # and asymmetric risk/reward.  Combines signal strength, conviction,
    # payoff asymmetry, and factor alignment into a single 0-100 score.
    df["sleeve_rank_agreement"] = (
        (pct_score(df["rsi_value_score"], lower_is_better=False).fillna(0)
         + pct_score(df["squeeze_laggard_score"], lower_is_better=False).fillna(0)
         + pct_score(df["value_laggard_score"], lower_is_better=False).fillna(0)
         + pct_score(df["momentum_pullback_score"], lower_is_better=False).fillna(0))
        / 4.0  # average percentile across all 4 sleeves
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

    score_cols = ["rsi_value_score", "squeeze_laggard_score", "value_laggard_score", "momentum_pullback_score"]
    labels = {
        "rsi_value_score": "RSI inflection + value",
        "squeeze_laggard_score": "shorted near lows / peer lag",
        "value_laggard_score": "cheap peer laggard",
        "momentum_pullback_score": "momentum pullback",
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


def _clean_company_for_query(company: str | None) -> str:
    if not company:
        return ""
    text = re.sub(r"\b(Common Stock|Class [A-Z]|Inc\.?|Corporation|Corp\.?|Ltd\.?|PLC|Company|Co\.?)\b", " ", str(company), flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _compact_text(text: str, max_chars: int = 1600) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    # Trim common boilerplate tails before they overwhelm actual commentary.
    for marker in ["Terms of Use", "Privacy Policy", "Cookie Policy", "All rights reserved"]:
        idx = text.lower().find(marker.lower())
        if idx > 700:
            text = text[:idx]
            break
    return text[:max_chars].strip()


def _is_bad_commentary_url(result_url: str) -> bool:
    parsed = urllib.parse.urlparse(result_url)
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    bad_domains = ("finviz.com", "barchart.com", "stockcharts.com", "google.com", "bing.com", "duckduckgo.com")
    bad_paths = ("/quote/", "/symbol/", "/market-data/quotes/", "/research-ratings", "/analysis/", "/analyst-ratings/", "expert-time")
    if any(bad in domain for bad in bad_domains):
        return True
    if domain == "finance.yahoo.com" and not (path.startswith("/news/") or path.startswith("/markets/") or path.startswith("/sectors/")):
        return True
    if domain == "seekingalpha.com" and (path.startswith("/symbol/") or path.endswith("/earnings")):
        return True
    return any(bad in path for bad in bad_paths)


def yahoo_finance_news_search(ticker: str, max_results: int = 8) -> list[dict]:
    url = "https://query1.finance.yahoo.com/v1/finance/search?" + urllib.parse.urlencode({
        "q": ticker,
        "quotesCount": 0,
        "newsCount": max_results,
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []
    out = []
    for news in payload.get("news", []):
        link = str(news.get("link") or "").strip()
        title = _compact_text(str(news.get("title") or ""), 180)
        if not link.startswith(("http://", "https://")) or _is_bad_commentary_url(link):
            continue
        title_l = title.lower()
        if ticker.lower() not in title_l and "$" + ticker.lower() not in title_l:
            # Keep Yahoo's broad ticker feed honest; unrelated sector articles are usually not ticker-specific commentary.
            continue
        out.append({
            "title": title,
            "url": link,
            "snippet": _compact_text(str(news.get("summary") or news.get("publisher") or ""), 700),
            "query": f"Yahoo Finance news API: {ticker}",
        })
    return out


def tavily_commentary_search(ticker: str, company: str | None, max_results: int = 8) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    company_q = _clean_company_for_query(company)
    query = (
        f'{ticker} {company_q} stock latest earnings analyst commentary outlook '
        '-site:finance.yahoo.com/quote -site:seekingalpha.com/symbol'
    )
    body = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "topic": "news",
        "include_answer": False,
    }).encode("utf-8")
    try:
        req = urllib.request.Request("https://api.tavily.com/search", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []
    found = []
    for result in payload.get("results", []):
        result_url = str(result.get("url") or "").strip()
        if not result_url.startswith(("http://", "https://")) or _is_bad_commentary_url(result_url):
            continue
        found.append({
            "title": _compact_text(str(result.get("title") or ""), 180),
            "url": result_url,
            "snippet": _compact_text(str(result.get("content") or ""), 700),
            "query": query,
        })
    return found


def search_commentary_web(ticker: str, company: str | None, max_results: int = 8) -> list[dict]:
    """Search for external stock commentary URLs. Prefer ticker-specific Yahoo news, then Tavily, then self-hosted SearXNG."""
    found: list[dict] = []
    seen_urls: set[str] = set()
    for hit in yahoo_finance_news_search(ticker, max_results=max_results):
        if hit["url"] not in seen_urls:
            seen_urls.add(hit["url"])
            found.append(hit)
        if len(found) >= max_results:
            return found
    for hit in tavily_commentary_search(ticker, company, max_results=max_results):
        if hit["url"] not in seen_urls:
            seen_urls.add(hit["url"])
            found.append(hit)
        if len(found) >= max_results:
            return found

    searxng = os.environ.get("SEARXNG_URL", "http://localhost:8888").rstrip("/")
    company_q = _clean_company_for_query(company)
    queries = [
        f'"{ticker}" "{company_q}" stock news earnings analyst outlook -site:finance.yahoo.com/quote -site:seekingalpha.com/symbol',
        f'"{ticker}" "{company_q}" why shares stock earnings guidance',
        f'"{ticker}" "{company_q}" downgrade upgrade analyst stock outlook',
    ]
    for query in queries:
        params = urllib.parse.urlencode({"q": query, "format": "json", "language": "en-US"})
        url = f"{searxng}/search?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        for result in payload.get("results", []):
            result_url = str(result.get("url") or "").strip()
            if not result_url.startswith(("http://", "https://")) or result_url in seen_urls or _is_bad_commentary_url(result_url):
                continue
            seen_urls.add(result_url)
            found.append({
                "title": _compact_text(str(result.get("title") or ""), 180),
                "url": result_url,
                "snippet": _compact_text(str(result.get("content") or result.get("snippet") or ""), 700),
                "query": query,
            })
            if len(found) >= max_results:
                return found
    return found


def extract_commentary_source(source: dict, ticker: str, company: str | None) -> dict | None:
    """Fetch and extract readable text from a commentary URL. Falls back to search snippet if extraction fails."""
    url = source.get("url")
    if not url:
        return None
    text = ""
    title = source.get("title") or ""
    try:
        req = urllib.request.Request(str(url), headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 AppleWebKit/537.36 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(1_500_000).decode("utf-8", errors="replace")
        try:
            from readability import Document
            from html2text import html2text
            doc = Document(raw)
            title = title or doc.short_title() or doc.title()
            text = html2text(doc.summary())
        except Exception:
            raw = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw)
            text = re.sub(r"(?s)<[^>]+>", " ", raw)
    except Exception:
        text = source.get("snippet") or ""
    excerpt = _compact_text(text, 1800)
    snippet = _compact_text(source.get("snippet") or "", 450)
    company_token = (_clean_company_for_query(company).split(" ") or [""])[0].lower()
    haystack = f"{title} {snippet} {excerpt}".lower()
    ticker_re = re.compile(rf"(?<![a-z0-9])\$?{re.escape(ticker.lower())}(?![a-z0-9])")
    if not ticker_re.search(haystack) and (company_token and company_token not in haystack):
        return None
    if len(excerpt) < 240 and len(snippet) < 120:
        return None
    return {
        "title": _compact_text(title, 180) or "Untitled source",
        "url": str(url),
        "excerpt": excerpt if len(excerpt) >= 240 else snippet,
        "search_snippet": snippet,
    }


def collect_web_commentary(row: pd.Series, max_sources: int = 3) -> list[dict]:
    ticker = str(row["ticker"])
    company = row.get("company")
    sources = []
    for hit in search_commentary_web(ticker, company, max_results=10):
        extracted = extract_commentary_source(hit, ticker, company)
        if extracted:
            sources.append(extracted)
        if len(sources) >= max_sources:
            break
    return sources


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
        cache_key = f"web-v4:{ticker}:{row['ts0']}:{round(float(row['opportunity_score']), 2)}:{row.get('primary_strategy')}"
        cached = cache.get(ticker)
        if cached and cached.get("cache_key") == cache_key and cached.get("sources"):
            analyses.append(cached)
            continue
        payload = {
            "ticker": ticker,
            "company": row.get("company"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "primary_strategy": row.get("primary_strategy"),
            "price": round(float(row.get("display_close")), 4),
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
        web_sources = [] if (cached and cached.get("cache_key") == cache_key and cached.get("sources")) else collect_web_commentary(row, max_sources=3)
        source_block = "\n\n".join(
            f"[{i}] {src['title']}\nURL: {src['url']}\nExcerpt: {src['excerpt']}"
            for i, src in enumerate(web_sources, 1)
        ) or "No reliable external commentary sources were extracted."
        prompt = (
            "Analyze this candidate as a possible LONG setup for a research dashboard, but do NOT merely restate the RSI/price chart. "
            "Use the web-extracted source excerpts as the primary qualitative input. The warehouse metrics are only context. "
            "Cite external commentary inline as [1], [2], etc. If the source excerpts do not support a claim, say it is unknown. "
            "Return exactly 5 bullets with labels: External commentary, Why it can work, What can break it, Confirming evidence to watch, Bottom line. "
            "Be specific, skeptical, and compliance-safe. No trade recommendation, no target price.\n\n"
            "Warehouse context:\n" + json.dumps(payload, indent=2) + "\n\n"
            "Web-extracted commentary sources:\n" + source_block
        )
        text = call_llm(prompt) if web_sources else (
            "- **External commentary**: No reliable external commentary source was extracted by the web-search pipeline for this run; avoid treating this as a catalyst-backed setup.\n"
            "- **Why it can work**: Unknown from external commentary. Only deterministic warehouse factors are available.\n"
            "- **What can break it**: Unknown from external commentary; fundamental or news-driven risks require manual source review.\n"
            "- **Confirming evidence to watch**: Fresh earnings commentary, management guidance, analyst notes, or company filings that validate the setup.\n"
            "- **Bottom line**: Source coverage failed, so this card should be read as quantitatively flagged but qualitatively unverified."
        )
        item = {
            "ticker": ticker,
            "cache_key": cache_key,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input": payload,
            "sources": web_sources,
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
        "global_rank", "rank_in_sector", "sector", "ticker", "company", "market_cap", "four_h_close", "display_close", "price_source", "latest_daily_close",
        "opportunity_score", "rsi_value_score", "squeeze_laggard_score", "value_laggard_score", "momentum_pullback_score", "ev_score",
        "rsi_acceleration_score", "composite_value_score", "rsi0", "rsi_delta_1",
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
        if key in {"sector", "ticker", "company", "value_grade", "growth_grade", "momentum_grade", "primary_strategy", "production_factor_basket", "production_theme", "primary_keyword_factor", "keyword_factor_baskets"}:
            out[key] = None if pd.isna(value) else str(value)
        elif key in {"four_h_timestamp", "latest_daily_timestamp"}:
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


def factor_basket_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "production_factor_basket" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = df[df["production_factor_basket"].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = work.groupby("production_factor_basket", dropna=True)
    baskets = g.agg(
        ticker_count=("ticker", "count"),
        avg_opportunity_score=("opportunity_score", "mean"),
        avg_factor_score=("production_factor_score", "mean"),
        avg_value_score=("composite_value_score", "mean"),
        avg_ret_1w_pct=("ret_1w_pct", "mean"),
        avg_ret_1m_pct=("ret_1m_pct", "mean"),
        avg_ret_3m_pct=("ret_3m_pct", "mean"),
        avg_ret_ytd_pct=("ret_ytd_pct", "mean"),
        avg_rsi=("rsi0", "mean"),
        avg_rsi_delta_1=("rsi_delta_1", "mean"),
        avg_rsi_accel=("rsi_accel", "mean"),
        inflection_count=("is_top_inflection", "sum"),
    ).reset_index().rename(columns={"production_factor_basket": "basket_name"})
    baskets = baskets[baskets["ticker_count"] >= 3].copy()
    if baskets.empty:
        return baskets, pd.DataFrame()
    baskets["lag_score"] = (
        np.maximum(0, -baskets["avg_ret_1m_pct"].fillna(0))
        + 0.50 * np.maximum(0, -baskets["avg_ret_3m_pct"].fillna(0))
        + 0.25 * np.maximum(0, -baskets["avg_ret_ytd_pct"].fillna(0))
    )
    baskets["inflection_score"] = (
        4.0 * np.maximum(0, baskets["avg_rsi_delta_1"].fillna(0))
        + 2.0 * np.maximum(0, baskets["avg_rsi_accel"].fillna(0))
        + 0.5 * np.maximum(0, baskets["avg_ret_1w_pct"].fillna(0))
        + 3.0 * baskets["inflection_count"].fillna(0) / baskets["ticker_count"].clip(lower=1)
    )
    baskets["is_lagged"] = (
        (baskets["avg_ret_1m_pct"].fillna(0) < 0)
        | (baskets["avg_ret_3m_pct"].fillna(0) < 0)
        | (baskets["avg_ret_ytd_pct"].fillna(0) < 0)
    )
    baskets["factor_reversal_score"] = (
        0.60 * pct_score(baskets["lag_score"], lower_is_better=False).fillna(50)
        + 0.40 * pct_score(baskets["inflection_score"], lower_is_better=False).fillna(50)
    )
    # The selected factor must be a true laggard; non-lagged baskets stay visible but cannot win.
    baskets["display_score"] = np.where(baskets["is_lagged"], baskets["factor_reversal_score"], baskets["factor_reversal_score"] * 0.25)
    baskets = baskets.sort_values(["display_score", "factor_reversal_score"], ascending=False).reset_index(drop=True)
    baskets["rank"] = baskets.index + 1
    target_basket = baskets[baskets["is_lagged"]].iloc[0]["basket_name"] if baskets["is_lagged"].any() else baskets.iloc[0]["basket_name"]
    opps = work[work["production_factor_basket"] == target_basket].copy()
    opps["factor_opportunity_score"] = (
        0.35 * opps["rsi_value_score"].fillna(50)
        + 0.25 * opps["composite_value_score"].fillna(50)
        + 0.20 * opps["production_factor_score"].fillna(opps["production_factor_score"].median()) * 10.0
        + 0.20 * opps["peer_lag_score"].fillna(50)
    ).clip(0, 100)
    opps = cap_by_sector(opps.sort_values("factor_opportunity_score", ascending=False), "factor_opportunity_score", 20, 4)
    return baskets, opps


def keyword_theme_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank Polygon keyword/thematic baskets like AI Infrastructure, Cloud Software, Oil & Gas."""
    if df.empty or "primary_keyword_factor" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    work = df[df["primary_keyword_factor"].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = work.groupby("primary_keyword_factor", dropna=True)
    themes = g.agg(
        ticker_count=("ticker", "count"),
        avg_keyword_score=("primary_keyword_factor_score", "mean"),
        avg_opportunity_score=("opportunity_score", "mean"),
        avg_factor_score=("production_factor_score", "mean"),
        avg_value_score=("composite_value_score", "mean"),
        avg_ret_1w_pct=("ret_1w_pct", "mean"),
        avg_ret_1m_pct=("ret_1m_pct", "mean"),
        avg_ret_3m_pct=("ret_3m_pct", "mean"),
        avg_ret_ytd_pct=("ret_ytd_pct", "mean"),
        avg_rsi=("rsi0", "mean"),
        avg_rsi_delta_1=("rsi_delta_1", "mean"),
        avg_rsi_accel=("rsi_accel", "mean"),
        inflection_count=("is_top_inflection", "sum"),
    ).reset_index().rename(columns={"primary_keyword_factor": "theme_name"})
    themes = themes[themes["ticker_count"] >= 3].copy()
    if themes.empty:
        return themes, pd.DataFrame()
    themes["lag_score"] = (
        np.maximum(0, -themes["avg_ret_1m_pct"].fillna(0))
        + 0.50 * np.maximum(0, -themes["avg_ret_3m_pct"].fillna(0))
        + 0.25 * np.maximum(0, -themes["avg_ret_ytd_pct"].fillna(0))
    )
    themes["inflection_score"] = (
        4.0 * np.maximum(0, themes["avg_rsi_delta_1"].fillna(0))
        + 2.0 * np.maximum(0, themes["avg_rsi_accel"].fillna(0))
        + 0.5 * np.maximum(0, themes["avg_ret_1w_pct"].fillna(0))
        + 3.0 * themes["inflection_count"].fillna(0) / themes["ticker_count"].clip(lower=1)
    )
    themes["is_lagged"] = (
        (themes["avg_ret_1m_pct"].fillna(0) < 0)
        | (themes["avg_ret_3m_pct"].fillna(0) < 0)
        | (themes["avg_ret_ytd_pct"].fillna(0) < 0)
    )
    themes["theme_reversal_score"] = (
        0.50 * pct_score(themes["lag_score"], lower_is_better=False).fillna(50)
        + 0.30 * pct_score(themes["inflection_score"], lower_is_better=False).fillna(50)
        + 0.20 * pct_score(themes["avg_keyword_score"], lower_is_better=False).fillna(50)
    )
    themes["display_score"] = np.where(themes["is_lagged"], themes["theme_reversal_score"], themes["theme_reversal_score"] * 0.25)
    themes = themes.sort_values(["display_score", "theme_reversal_score"], ascending=False).reset_index(drop=True)
    themes["rank"] = themes.index + 1
    target_theme = themes[themes["is_lagged"]].iloc[0]["theme_name"] if themes["is_lagged"].any() else themes.iloc[0]["theme_name"]
    opps = work[work["primary_keyword_factor"] == target_theme].copy()
    keyword_fill = opps["primary_keyword_factor_score"].median() if opps["primary_keyword_factor_score"].notna().any() else 10.0
    opps["theme_opportunity_score"] = (
        0.30 * opps["rsi_value_score"].fillna(50)
        + 0.25 * opps["squeeze_laggard_score"].fillna(50)
        + 0.20 * opps["composite_value_score"].fillna(50)
        + 0.15 * opps["primary_keyword_factor_score"].fillna(keyword_fill).clip(0, 50) * 2.0
        + 0.10 * opps["peer_lag_score"].fillna(50)
    ).clip(0, 100)
    opps = cap_by_sector(opps.sort_values("theme_opportunity_score", ascending=False), "theme_opportunity_score", 20, 4)
    return themes, opps


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
    pullbacks = cap_by_sector(df[df["mom_pullback_eligible"]].sort_values("momentum_pullback_score", ascending=False), "momentum_pullback_score", 15, 3)
    master_ev = cap_by_sector(
        df[df["ev_master_eligible"]].sort_values("ev_score", ascending=False),
        "ev_score", 20, 3,
    )

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
        "momentum_pullbacks": [record(r) for _, r in pullbacks.iterrows()],
        "master_opportunities": [record(r) for _, r in master_ev.iterrows()],
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
            f"<td>{fmt_money(r['display_close'])}<small>{html.escape(str(r.get('price_source', '')))}</small></td>"
            f"<td>{fmt_bn(r['market_cap'])}</td>"
            f"<td>{render_bar(r['opportunity_score'])}</td>"
            f"<td>{render_bar(r.get('rsi_value_score'), 'hot')}</td>"
            f"<td>{render_bar(r.get('squeeze_laggard_score'), 'short')}</td>"
            f"<td>{render_bar(r.get('value_laggard_score'), 'value')}</td>"
            f"<td>{render_bar(r.get('momentum_pullback_score'), 'mom')}</td>"
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
    pullback_rows = [row_html(r) for _, r in pullbacks.iterrows()]
    master_rows = [row_html(r) for _, r in master_ev.iterrows()]

    analysis_cards = []
    for item in analyses:
        inp = item["input"]
        text = html.escape(item["analysis"]).replace("\n", "<br>")
        sources = item.get("sources") or []
        if sources:
            source_links = "".join(
                f"<li><a href='{html.escape(src['url'], quote=True)}' target='_blank' rel='noopener noreferrer'>[{i}] {html.escape(src.get('title') or src['url'])}</a></li>"
                for i, src in enumerate(sources, 1)
            )
            sources_html = f"<div class='sources'><b>Extracted web sources</b><ol>{source_links}</ol></div>"
        else:
            sources_html = "<div class='sources missing'><b>Extracted web sources</b><p>None extracted for this run.</p></div>"
        analysis_cards.append(
            f"<article class='card'><div class='card-head'><span>{ticker_link(item['ticker'])}</span>"
            f"<em>{html.escape(str(inp.get('sector')))} · ${inp.get('price')}</em></div>"
            f"<div class='metrics'>{html.escape(str(inp.get('primary_strategy')))} · Score {inp.get('opportunity_score')} · short {inp.get('short_pct_float')}% · peer lag {inp.get('peer_lag_1m_pct')}%</div>"
            f"{sources_html}<p>{text}</p></article>"
        )

    sector_sections = []
    for sector, sdf in top_sector.groupby("sector"):
        items = []
        for _, r in sdf.iterrows():
            items.append(
                f"<li><b>{ticker_link(r['ticker'])}</b> {fmt_money(r['display_close'])} "
                f"score {fmt_num(r['opportunity_score'])} · {html.escape(str(r.get('primary_strategy')))} · short {fmt_num(r.get('short_pct_float'))}% · lag {fmt_num(r.get('peer_lag_1m_pct'))}%</li>"
            )
        sector_sections.append(f"<section class='sector'><h3>{html.escape(str(sector))}</h3><ol>{''.join(items)}</ol></section>")

    header = "<table><thead><tr><th>#</th><th>Ticker</th><th>Sector / Sleeve</th><th>4h Px</th><th>MCap</th><th>Opp</th><th>RSI</th><th>Short/Lows</th><th>Value/Lag</th><th>Momo Pb</th><th>Short</th><th>From Low</th><th>Peer Lag 1M</th><th>RSI Accel</th></tr></thead><tbody>"

    factor_baskets, factor_opps = factor_basket_analysis(df)
    factor_rows = []
    selected_basket = "none"
    if not factor_baskets.empty:
        selected_basket = str(factor_baskets.iloc[0]["basket_name"])
        for _, b in factor_baskets.iterrows():
            cls = " class='selected'" if str(b["basket_name"]) == selected_basket else ""
            factor_rows.append(
                f"<tr{cls}><td>{int(b['rank'])}</td><td><strong>{html.escape(str(b['basket_name']))}</strong><small>{int(b['ticker_count'])} names under ${price_filter:.0f}</small></td>"
                f"<td>{render_bar(b['factor_reversal_score'])}</td><td>{fmt_num(b['avg_ret_1w_pct'])}%</td><td>{fmt_num(b['avg_ret_1m_pct'])}%</td><td>{fmt_num(b['avg_ret_3m_pct'])}%</td>"
                f"<td>{fmt_num(b['avg_rsi'])}</td><td>{fmt_num(b['avg_rsi_delta_1'])}</td><td>{fmt_num(b['avg_rsi_accel'])}</td><td>{int(b['inflection_count'])}</td></tr>"
            )
    factor_opp_rows = []
    if not factor_opps.empty:
        for display_rank, (_, r) in enumerate(factor_opps.iterrows(), 1):
            rr = r.copy()
            rr["display_rank"] = display_rank
            factor_opp_rows.append(row_html(rr, "display_rank"))

    theme_baskets, theme_opps = keyword_theme_analysis(df)
    theme_rows = []
    selected_theme = "none"
    if not theme_baskets.empty:
        selected_theme = str(theme_baskets.iloc[0]["theme_name"])
        for _, b in theme_baskets.iterrows():
            cls = " class='selected'" if str(b["theme_name"]) == selected_theme else ""
            theme_rows.append(
                f"<tr{cls}><td>{int(b['rank'])}</td><td><strong>{html.escape(str(b['theme_name']))}</strong><small>{int(b['ticker_count'])} names under ${price_filter:.0f}</small></td>"
                f"<td>{render_bar(b['theme_reversal_score'])}</td><td>{fmt_num(b['avg_keyword_score'])}</td><td>{fmt_num(b['avg_ret_1w_pct'])}%</td><td>{fmt_num(b['avg_ret_1m_pct'])}%</td><td>{fmt_num(b['avg_ret_3m_pct'])}%</td>"
                f"<td>{fmt_num(b['avg_rsi'])}</td><td>{fmt_num(b['avg_rsi_delta_1'])}</td><td>{fmt_num(b['avg_rsi_accel'])}</td><td>{int(b['inflection_count'])}</td></tr>"
            )
    theme_opp_rows = []
    if not theme_opps.empty:
        for display_rank, (_, r) in enumerate(theme_opps.iterrows(), 1):
            rr = r.copy()
            rr["display_rank"] = display_rank
            theme_opp_rows.append(row_html(rr, "display_rank"))

    css = """
:root{--bg:#080808;--panel:#101010;--panel2:#151515;--text:#e8e4d8;--muted:#8a867b;--line:#2a2822;--amber:#e6b422;--green:#40c463;--red:#ff5c5c;--purple:#b58cff;--cyan:#47d7ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:'JetBrains Mono','SFMono-Regular',Consolas,monospace;font-size:13px}header{border-bottom:1px solid var(--line);padding:18px 22px;background:#0d0d0d;position:sticky;top:0;z-index:2}h1{font-size:18px;margin:0 0 8px;color:var(--amber);letter-spacing:.04em}h2{font-size:15px;margin:28px 0 12px;color:var(--amber);text-transform:uppercase}h3{font-size:13px;color:var(--amber)}.status{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted)}.dot{color:var(--green)}main{padding:18px 22px;max-width:1600px;margin:0 auto}.note{border:1px solid var(--line);padding:12px;background:var(--panel);color:var(--muted);line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}.card,.sector{border:1px solid var(--line);background:var(--panel);padding:12px}.card-head{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:8px}.card-head span{color:var(--amber);font-size:16px;font-weight:bold}.card-head em{font-style:normal;color:var(--muted)}.metrics{color:var(--green);margin-bottom:8px}.sources{border:1px solid var(--line);background:#0b0b0b;padding:8px;margin:8px 0;color:var(--muted)}.sources b{color:var(--cyan)}.sources ol{margin:6px 0 0 18px;padding:0}.sources li{margin:3px 0}.sources a{color:var(--amber);text-decoration:none}.sources a:hover{color:var(--green)}.sources.missing{border-color:#3a2a2a}.card p{line-height:1.55;margin:0;color:#d8d2c3}table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{border:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{color:var(--amber);font-weight:600;background:#0e0e0e;position:sticky;top:76px}td small{display:block;color:var(--muted);font-size:11px;margin-top:3px}a.ticker-link{color:var(--amber);text-decoration:none;border-bottom:1px solid rgba(230,180,34,.45)}a.ticker-link:hover{color:var(--green);border-bottom-color:var(--green)}.nav{display:flex;gap:10px;margin-top:10px}.nav a{border:1px solid var(--line);padding:6px 8px;text-decoration:none}.nav a.active{color:#080808;background:var(--amber)}tr.selected td{background:#171203}.bar{display:inline-block;width:74px;height:7px;background:#242018;margin-right:8px;vertical-align:middle}.bar i{display:block;height:100%;background:var(--amber)}.bar.hot i{background:var(--green)}.bar.value i{background:var(--purple)}.bar.short i{background:var(--cyan)}.sector ol{margin:0;padding-left:22px}.sector li{margin:6px 0;line-height:1.45}.footer{color:var(--muted);font-size:11px;margin:28px 0}.pill{border:1px solid var(--line);padding:3px 6px;color:var(--amber);display:inline-block;margin-right:6px}a{color:var(--amber)}
.tab-nav{display:flex;flex-wrap:wrap;gap:2px;border-bottom:2px solid var(--amber);margin-bottom:18px;position:sticky;top:90px;z-index:1;background:var(--bg);padding:4px 0}
.tab-nav input[type=radio]{display:none}
.tab-nav label{display:inline-block;padding:8px 14px;border:1px solid var(--line);border-bottom:none;background:var(--panel);color:var(--muted);cursor:pointer;font-size:13px;font-weight:600;border-radius:4px 4px 0 0;margin-right:2px;transition:all 0.15s}
.tab-nav label:hover{color:var(--amber);background:var(--panel2)}
.tab-nav input:checked+label{background:var(--amber);color:#0a0a0a;border-color:var(--amber)}
.tab-content{display:none}
#tab-opps:checked~.tab-nav label.opps-tab,#tab-rsi:checked~.tab-nav label.rsi-tab,#tab-sqz:checked~.tab-nav label.sqz-tab,#tab-val:checked~.tab-nav label.val-tab,#tab-mom:checked~.tab-nav label.mom-tab,#tab-master:checked~.tab-nav label.master-tab,#tab-sector:checked~.tab-nav label.sector-tab{background:var(--amber);color:#0a0a0a;border-color:var(--amber)}
#tab-opps:checked~#c-opps,#tab-rsi:checked~#c-rsi,#tab-sqz:checked~#c-sqz,#tab-val:checked~#c-val,#tab-mom:checked~#c-mom,#tab-master:checked~#c-master,#tab-sector:checked~#c-sector{display:block}
h2{margin-top:0}
"""
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity Screener</title><style>{css}</style></head>
<body><header><h1>EQUITY SCREENER</h1><div class="status"><span class="dot">● LIVE</span><span>generated {html.escape(now_et.strftime('%Y-%m-%d %H:%M %Z'))}</span><span>latest 4h {html.escape(latest_ts)}</span><span>price &lt; ${price_filter:.0f}</span><span>{len(df)} names / {df['sector'].nunique() if not df.empty else 0} sectors</span><span>top 10 capped at max 3/sector</span></div><nav class="nav"><a class="active" href="index.html">Opportunities</a><a href="factor-baskets.html">Factor basket inflections</a></nav></header>
<main>
<div class="note"><span class="pill">Method</span> Multi-sleeve rank: RSI inflection + value, shorted-near-lows / peer lag, cheap peer laggards, and momentum pullbacks. The top 10 blends four sleeves and is diversified with a hard cap of 3 stocks per sector. Momentum pullbacks scan for strong 6-month uptrends that have pulled back 1-2 weeks and are coiling into moving averages — a continuation setup.</div>

<!-- Tab radio inputs -->
<input type="radio" name="tab" id="tab-opps" checked>
<input type="radio" name="tab" id="tab-rsi">
<input type="radio" name="tab" id="tab-sqz">
<input type="radio" name="tab" id="tab-val">
<input type="radio" name="tab" id="tab-mom">
<input type="radio" name="tab" id="tab-master">
<input type="radio" name="tab" id="tab-sector">

<nav class="tab-nav">
<label class="opps-tab" for="tab-opps">🥇 Opportunities</label>
<label class="rsi-tab" for="tab-rsi">📈 RSI Inflections</label>
<label class="sqz-tab" for="tab-sqz">🔻 Squeeze Laggards</label>
<label class="val-tab" for="tab-val">💰 Value Laggards</label>
<label class="mom-tab" for="tab-mom">🚀 Momentum Pullbacks</label>
<label class="master-tab" for="tab-master">⭐ Master Opportunities</label>
<label class="sector-tab" for="tab-sector">🏭 By Sector</label>
</nav>

<div class="tab-content" id="c-opps">
<h2>Multi-sleeve top 10 opportunities</h2>{header}{''.join(div_rows)}</tbody></table>
<h2>Web-sourced deep dive: diversified top 10</h2><div class="grid">{''.join(analysis_cards)}</div>
<h2>Top 25 diversified ranked opportunities</h2>{header}{''.join(top_rows)}</tbody></table>
</div>

<div class="tab-content" id="c-rsi">
<h2>RSI inflection sleeve</h2>{header}{''.join(inflect_rows)}</tbody></table>
</div>

<div class="tab-content" id="c-sqz">
<h2>Shorted near lows / peer lag sleeve</h2>{header}{''.join(squeeze_rows)}</tbody></table>
</div>

<div class="tab-content" id="c-val">
<h2>Cheap peer laggard sleeve</h2>{header}{''.join(laggard_rows)}</tbody></table>
</div>

<div class="tab-content" id="c-mom">
<h2>Momentum pullback sleeve</h2>{header}{''.join(pullback_rows)}</tbody></table>
</div>

<div class="tab-content" id="c-master">
<h2>⭐ Master Opportunities — High Expected Value</h2><p class="note"><span class="pill">EV Formula</span> 35% top sleeve signal + 25% cross-sleeve agreement + 25% asymmetric R:R + 15% factor alignment. Only stocks scoring ≥60 across all dimensions qualify. Sorted by EV score, capped at 3 per sector.</p>{header}{''.join(master_rows)}</tbody></table>
</div>

<div class="tab-content" id="c-sector">
<h2>Top by sector</h2><div class="grid">{''.join(sector_sections)}</div>
</div>

<div class="footer">Known: numeric data from local Polygon/DuckDB warehouse; cited qualitative commentary from extracted web sources. Estimated: composite scores from normalized warehouse fields. Unknown: catalysts or risks not present in extracted sources or warehouse fields.</div>
</main></body></html>"""
    factor_header = "<table><thead><tr><th>#</th><th>Factor basket</th><th>Reversal</th><th>1W</th><th>1M</th><th>3M</th><th>RSI</th><th>RSI Δ1</th><th>RSI Accel</th><th>Inflect Names</th></tr></thead><tbody>"
    theme_header = "<table><thead><tr><th>#</th><th>Keyword / theme basket</th><th>Reversal</th><th>Theme Score</th><th>1W</th><th>1M</th><th>3M</th><th>RSI</th><th>RSI Δ1</th><th>RSI Accel</th><th>Inflect Names</th></tr></thead><tbody>"
    factor_content = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Factor Basket Inflections</title><style>{css}</style></head>
<body><header><h1>FACTOR + KEYWORD BASKET LAGGARDS / INFLECTIONS</h1><div class="status"><span class="dot">● LIVE</span><span>generated {html.escape(now_et.strftime('%Y-%m-%d %H:%M %Z'))}</span><span>price &lt; ${price_filter:.0f}</span><span>selected factor: {html.escape(selected_basket)}</span><span>selected theme: {html.escape(selected_theme)}</span></div><nav class="nav"><a href="index.html">Opportunities</a><a class="active" href="factor-baskets.html">Factor basket inflections</a></nav></header>
<main>
<div class="note"><span class="pill">Method</span> First rank production factor baskets, then separately rank Polygon keyword/theme baskets such as AI Infrastructure, Cloud Software, Defense / Aerospace, and other beneficiary groups. Both use lag plus latest 4h RSI/return inflection. Then show the best sub-${price_filter:.0f} names inside the highest-ranked lagging factor and keyword/theme baskets.</div>
<h2>Production factor basket score + momentum analysis</h2>{factor_header}{''.join(factor_rows)}</tbody></table>
<h2>Best opportunities within selected lagging / inflecting production factor: {html.escape(selected_basket)}</h2>{header}{''.join(factor_opp_rows)}</tbody></table>
<h2>Keyword / theme basket score + momentum analysis</h2>{theme_header}{''.join(theme_rows)}</tbody></table>
<h2>Best opportunities within selected lagging / inflecting keyword theme: {html.escape(selected_theme)}</h2>{header}{''.join(theme_opp_rows)}</tbody></table>
<div class="footer">Known: production factor baskets, primary keyword factors, prices, technicals, returns, and factor scores from local Polygon/DuckDB warehouse. Estimated: reversal scores are deterministic composites of basket lag, keyword relevance, and short-term inflection.</div>
</main></body></html>"""
    (DOCS_DIR / "factor-baskets.html").write_text(factor_content)
    (DOCS_DIR / "index.html").write_text(content)

def git_commit_push() -> None:
    subprocess.run(["git", "add", "README.md", ".gitignore", "run_daily.sh", "scripts/build_dashboard.py", "docs/index.html", "docs/factor-baskets.html", "docs/dashboard_data.json", "data/dashboard_data.json", "data/llm_analysis.json", "data/scored_candidates.csv"], cwd=PROJECT_DIR, check=True)
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
    parser.add_argument("--price-filter", type=float, default=50.0)
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
