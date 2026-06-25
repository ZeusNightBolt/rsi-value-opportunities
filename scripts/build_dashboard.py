#!/usr/bin/env python3
"""Compatibility entrypoint for the modular equity screener build package."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from equity_screener.baskets import factor_basket_analysis, keyword_theme_analysis
from equity_screener.commentary import analyze_top_inflections
from equity_screener.config import *
from equity_screener.data import load_env_file, query_candidates
from equity_screener.git_ops import git_commit_push
from equity_screener.main import main
from equity_screener.polygon_overlay import enrich_latest_polygon_prices
from equity_screener.render import render_dashboard
from equity_screener.render_helpers import *
from equity_screener.scoring import grade_to_score, pct_score, score_candidates
from equity_screener.selection import build_diversified_top10, cap_by_sector, final_candidate_tickers, mark_diversified_top10
from equity_screener.serialization import clean_float, record


if __name__ == "__main__":
    raise SystemExit(main())
