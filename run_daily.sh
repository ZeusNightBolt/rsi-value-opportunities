#!/usr/bin/env bash
# Equity Screener — daily build + deploy
# 1. Refreshes warehouse data (latest hourly + indicators)
# 2. Builds dashboard
# 3. Pushes to GitHub Pages
set -euo pipefail

LOG_DIR="/home/nima/market-data/logs"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
LOG_FILE="$LOG_DIR/equity_screener_${RUN_ID}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "═══════════════════════════════════════"
echo "  EQUITY SCREENER RUN"
echo "  Run: $RUN_ID"
echo "  $(date -Is)"
echo "═══════════════════════════════════════"

# Step 1: Refresh warehouse data (latest 1h/4h data points)
echo ""
echo "── Step 1: Warehouse quick refresh ──"
if /usr/bin/bash /home/nima/market-data/scripts/quick_refresh.sh; then
    echo "   ✅ Warehouse refresh complete"
else
    rc=$?
    echo "   ⚠️  Warehouse refresh failed (exit $rc) — building with existing data"
fi

# Step 2: Freshness guard
echo ""
echo "── Step 2: Data freshness check ──"
/usr/bin/python3 -c "
import duckdb, sys, datetime
db = duckdb.connect('/home/nima/market-data/market_data.duckdb', read_only=True)
ti_max = db.execute(\"SELECT max(to_timestamp(CAST(timestamp/1000 AS BIGINT))) FROM technical_indicators WHERE timeframe='4h'\").fetchone()
if ti_max[0] is None:
    print('FATAL: no 4h RSI data')
    sys.exit(2)
age_hours = (datetime.datetime.now(datetime.timezone.utc) - ti_max[0].replace(tzinfo=datetime.timezone.utc)).total_seconds() / 3600
if age_hours > 30:
    print(f'REFUSING: 4h RSI data is {age_hours:.0f}h old')
    sys.exit(2)
print(f'OK: 4h RSI data is {age_hours:.0f}h old')
db.close()
" || exit $?

# Step 3: Build + deploy dashboard
echo ""
echo "── Step 3: Build + deploy dashboard ──"
cd /home/nima/rsi-value-opportunities
/usr/bin/python3 scripts/build_dashboard.py --price-filter 75 --top-llm 10 --push

echo ""
echo "═══════════════════════════════════════"
echo "  EQUITY SCREENER COMPLETE"
echo "  $(date -Is)"
echo "  Log: $LOG_FILE"
echo "═══════════════════════════════════════"
