from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = Path.home() / "market-data" / "market_data.duckdb"
ENV_PATH = Path.home() / ".hermes" / ".env"
MARKET_DATA_DIR = Path.home() / "market-data"
DOCS_DIR = PROJECT_DIR / "docs"
DATA_DIR = PROJECT_DIR / "data"

SCORE_COLUMNS = [
    "rsi_value_score",
    "squeeze_laggard_score",
    "value_laggard_score",
    "momentum_leader_score",
    "momentum_pullback_score",
    "rel_strength_pullback_score",
    "inflect_breakout_score",
]

SLEEVE_LABELS = {
    "rsi_value_score": "RSI inflection + value",
    "squeeze_laggard_score": "shorted near lows / peer lag",
    "value_laggard_score": "cheap peer laggard",
    "momentum_leader_score": "momentum leader / wave 3",
    "momentum_pullback_score": "momentum pullback / wave 2-4",
    "rel_strength_pullback_score": "relative strength pullback",
    "inflect_breakout_score": "RSI breakout inflection",
    "wave_setup_score": "wave-stage setup",
}

SCORE_DISPLAY = [
    ("Opp", "opportunity_score", ""),
    ("RSI", "rsi_value_score", "hot"),
    ("Sqz", "squeeze_laggard_score", "short"),
    ("Val", "value_laggard_score", "value"),
    ("Lead", "momentum_leader_score", "lead"),
    ("Momo", "momentum_pullback_score", "mom"),
    ("RS", "rel_strength_pullback_score", "rs"),
    ("Brk", "inflect_breakout_score", "brk"),
    ("Wave", "wave_setup_score", "wave"),
]

DIVERSIFIED_TOP_PLAN = [
    ("opportunity_score", 1),
    ("rsi_value_score", 1),
    ("squeeze_laggard_score", 1),
    ("value_laggard_score", 1),
    ("momentum_leader_score", 2),
    ("momentum_pullback_score", 1),
    ("rel_strength_pullback_score", 1),
    ("inflect_breakout_score", 1),
    ("wave_setup_score", 1),
]

COLOR_RGB = {
    "hot": "64,196,99",
    "value": "181,140,255",
    "short": "71,215,255",
    "mom": "255,92,92",
    "lead": "122,167,255",
    "rs": "255,165,0",
    "brk": "0,206,209",
    "wave": "66,214,140",
}

GIT_TRACKED_OUTPUTS = [
    "README.md",
    ".gitignore",
    "run_daily.sh",
    "scripts/build_dashboard.py",
    "scripts/equity_screener/__init__.py",
    "scripts/equity_screener/baskets.py",
    "scripts/equity_screener/commentary.py",
    "scripts/equity_screener/combined.py",
    "scripts/equity_screener/freshness.py",
    "scripts/equity_screener/config.py",
    "scripts/equity_screener/data.py",
    "scripts/equity_screener/git_ops.py",
    "scripts/equity_screener/main.py",
    "scripts/equity_screener/polygon_overlay.py",
    "scripts/equity_screener/render.py",
    "scripts/equity_screener/render_helpers.py",
    "scripts/equity_screener/scoring.py",
    "scripts/equity_screener/selection.py",
    "scripts/equity_screener/serialization.py",
    "docs/index.html",
    "docs/factor-baskets.html",
    "docs/divergence.html",
    "docs/universe.html",
    "docs/_config.yml",
    "docs/_layouts/default.html",
    "docs/dashboard_data.json",
    "data/dashboard_data.json",
    "data/llm_analysis.json",
    "data/scored_candidates.csv",
    "test_build_dashboard.py",
    "test_freshness_guard.py",
]
