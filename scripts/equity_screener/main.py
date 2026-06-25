import argparse
import json

from .commentary import analyze_top_inflections
from .config import DATA_DIR, DOCS_DIR, ENV_PATH
from .data import load_env_file, query_candidates
from .git_ops import git_commit_push
from .polygon_overlay import enrich_latest_polygon_prices
from .render import render_dashboard
from .scoring import score_candidates
from .selection import final_candidate_tickers, mark_diversified_top10

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
    df = mark_diversified_top10(score_candidates(df))
    final_tickers = final_candidate_tickers(df)
    df = enrich_latest_polygon_prices(df, final_tickers)
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
