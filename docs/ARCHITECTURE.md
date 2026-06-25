# Equity Screener Architecture and Audit

_Last refreshed: 2026-06-25_

## Objective

Equity Screener is a static GitHub Pages dashboard that ranks low-priced, $5B+ market-cap U.S. equities from the local Polygon/DuckDB warehouse. It is a research dashboard only; numeric fields come from local warehouse tables and qualitative commentary is downstream/contextual.

## Runtime entry points

| Entry point | Purpose |
|---|---|
| `run_daily.sh` | Cron wrapper: quick warehouse refresh, 4h indicator freshness guard, then dashboard build/deploy. |
| `scripts/build_dashboard.py` | Main pipeline: query DuckDB, score candidates, optional LLM/web commentary, render HTML/JSON/CSV, optionally commit/push. |
| `docs/index.html` | GitHub Pages main dashboard output. |
| `docs/factor-baskets.html` | GitHub Pages factor/theme basket output. |
| `docs/dashboard_data.json` | Static JSON payload used for audit/API consumers. |
| `data/scored_candidates.csv` | Local full scored universe output. |
| `data/llm_analysis.json` | Cache of LLM commentary for diversified top names. |

## Data flow

```text
~/market-data/market_data.duckdb
  ├─ technical_indicators, timeframe='4h'        → RSI history / acceleration
  ├─ daily_bars                                  → latest daily close + 52w range
  ├─ v_vti_sector_universe_5b                    → sector, market cap, short/volume context
  ├─ vti_daily_enriched_latest                   → valuation grades and multiples
  └─ v_vti_factor_production_scores_5b           → factor basket/theme/returns
       ↓
scripts/build_dashboard.py
  ├─ query_candidates(price_filter)
  ├─ score_candidates(df)
  ├─ final_candidate_tickers(df)                   dashboard-surface union only
  ├─ enrich_latest_polygon_prices(df, tickers)     live Polygon snapshot overlay
  ├─ analyze_top_inflections(df, top_n, force)    optional LLM/web commentary
  ├─ render_dashboard(df, analyses, price_filter)
  └─ git_commit_push()                            only when --push is set
       ↓
docs/*.html + docs/dashboard_data.json + data/*.json/csv
```

## Scoring model

`SCORE_COLUMNS` is the canonical six-sleeve list. Add future sleeves there first, then add label/display metadata to `SLEEVE_LABELS` and `SCORE_DISPLAY`.

| Sleeve | Score column | Intent |
|---|---|---|
| RSI inflection + value | `rsi_value_score` | Mean-reversion/value names where 4h RSI is accelerating upward. |
| Squeeze laggard | `squeeze_laggard_score` | Shorted names near lows that lag sector peers. |
| Value laggard | `value_laggard_score` | Cheap names lagging sector peers. |
| Momentum pullback | `momentum_pullback_score` | Relative 6-month winners that pulled back and are coiling near SMAs. |
| Relative strength pullback | `rel_strength_pullback_score` | Strong stocks in non-broken sectors with recent pullbacks into support. |
| RSI breakout / inflection | `inflect_breakout_score` | RSI turning up from 40-60 with volume expansion and sector tailwind. |

`opportunity_score` is `max(SCORE_COLUMNS)`. `ev_score` combines top-sleeve signal, average sleeve percentile agreement, 52-week payoff asymmetry, and factor alignment.

## Rendering model

The dashboard is still a single-file renderer, but the repeated score metadata is centralized. Prices shown for final dashboard candidates are overlaid at build time from Polygon snapshot data after deterministic scoring. The warehouse-derived `display_close` is preserved as `warehouse_display_close`; the rendered `display_close` and `price_source` use `polygon.snapshot.lastTrade.p`, `polygon.snapshot.min.c`, `polygon.snapshot.day.c`, or `polygon.snapshot.prevDay.c` when Polygon returns a usable snapshot price.

Centralized metadata:

- `SCORE_COLUMNS`: canonical scoring contract.
- `DIVERSIFIED_TOP_PLAN`: per-sleeve quotas for diversified top 10 construction; selected rows carry `diversified_source` in the JSON payload.
- `SLEEVE_LABELS`: maps score column to human strategy label.
- `SCORE_DISPLAY`: mobile card and master-comparison display rows.
- `COLOR_RGB`: score heatmap colors.
- `GIT_TRACKED_OUTPUTS`: files committed by `--push`.

The June 25 full revamp changed the generated surface from a dense table-first page to a trader cockpit:

- sticky hero with live freshness / price-filter state;
- left navigation rail plus horizontally scrollable section chips;
- KPI strip for universe, live Polygon price overlays, average opportunity score, average RSI, and commentary count;
- card-first top opportunities on desktop and mobile, with tables retained as audit trails;
- separate factor/theme map page using the same shell;
- Jekyll/GitHub Pages scaffold under `docs/_config.yml` and `docs/_layouts/default.html` so future static docs can use layout/theme conventions instead of ad hoc pages.

This is the first cleanup step toward a fully modular package. Next natural split:

```text
scripts/build_dashboard.py        thin CLI/orchestrator
src/equity_screener/config.py     score/tab metadata
src/equity_screener/data.py       DuckDB queries
src/equity_screener/scoring.py    sleeve scoring
src/equity_screener/render.py     HTML/JSON rendering
src/equity_screener/deploy.py     git commit/push
```

## Audit findings fixed in this cleanup

| Severity | Finding | Fix |
|---|---|---|
| High | `build_diversified_top10()` used cumulative quota values in a way that filled the top 10 before momentum/RS-breakout sleeves could contribute. | Added `DIVERSIFIED_TOP_PLAN` and made `add_from()` enforce per-sleeve quotas. |
| High | `git_commit_push()` always pushed `origin main`, which would fail to publish commits made on any non-main branch and violates branch-first repo hygiene. | Pushes the current checked-out branch now. |
| Medium | `score_candidates()` mutated the input DataFrame in place, making downstream tests/refactors harder to reason about. | It now returns a scored copy. |
| Medium | `record()` omitted `rel_strength_pullback_score` and treated `price_source` as numeric, so JSON payloads lost both fields. | Added payload contract tests; `record()` now preserves `price_source` and RS pullback score. |
| Medium | `diversified_top10` CSV flag used a different selection algorithm/index basis than the dashboard top-10 builder. | It is now computed from `build_diversified_top10()` after the final sort/reset. |
| Medium | `call_llm()` had narrow exception handling for network/SSL edge cases. | It now fails soft on any LLM exception and returns a diagnostic string. |
| Medium | Final dashboard rows could display stale warehouse 4h/daily fallback prices even though Polygon snapshot data was available at build time. | Added `final_candidate_tickers()` + `enrich_latest_polygon_prices()` and payload fields for latest Polygon snapshot price/status/source. |
| High | Dashboard was table-first with weak information hierarchy and slow navigation across nine opportunity surfaces. | Full revamp to sticky hero, left rail, KPI strip, card-first opportunity cockpit, scrollable section chips, and matching factor/theme map shell. |
| Medium | Mobile cards omitted the Relative Strength Pullback score even though desktop/master views showed it. | Mobile score rows are generated dynamically from `SCORE_DISPLAY`, including `RS Pb`. |
| Medium | Master opportunity comparison hard-coded six sleeve rows; future sleeves would require multiple edits. | Master comparison rows are generated from `SCORE_DISPLAY[1:]`. |
| Medium | Dashboard note said the top 10 blended five sleeves; the model actually uses six. | Note now says six sleeves. |
| Medium | RS Pullback tab described stale gate thresholds (`sector > 0`, `pullback < -3`, `>30% above low`) that no longer matched code. | Text now matches the current relaxed gate. |
| Low | Unused dead helpers `score_heatmap`, `render_mobile_card`, and `mobile_section`. | Removed. |
| Low | Repeated score-column lists and label maps created drift risk. | Centralized in module-level constants. |

## Remaining technical debt

- `scripts/build_dashboard.py` is still a large monolith (~1.5k lines). The next cleanup should split data, scoring, rendering, and deployment into modules under `src/equity_screener/`.
- Search/extraction failures intentionally degrade silently for dashboard uptime. If qualitative commentary quality becomes important, replace broad `except Exception: return []` with structured warnings in the JSON payload.
- The generated HTML is still string-template heavy. A small template file or Jinja-free renderer module would make tab additions safer.
- No formal pytest suite exists. Current verification is command-based: syntax check, full no-LLM build, JSON payload counts, HTML marker checks.

## Verification commands

Run these from `~/equity-screener`:

```bash
/usr/bin/python3 -m py_compile scripts/build_dashboard.py
/usr/bin/python3 scripts/build_dashboard.py --price-filter 75 --no-llm
/usr/bin/python3 - <<'PY'
import json, pathlib
payload = json.load(open('docs/dashboard_data.json'))
assert payload['universe_count'] > 0
for key in ['top_diversified','top','squeeze_laggards','value_laggards','momentum_pullbacks','inflect_breakouts','master_opportunities']:
    assert len(payload[key]) > 0, key
html = pathlib.Path('docs/index.html').read_text()
assert '{{D_DATA}}' not in html
assert 'RS Pb' in html
assert 'blends six sleeves' in html
assert 'pullback < -3%' not in html
print('ok')
PY
```
