from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

RETRY_STALE_HOURS = 50
WEEKDAY_MAX_STALE_HOURS = 30
WEEKEND_MAX_STALE_HOURS = 80
EXIT_RETRY_REFRESH = 75


@dataclasses.dataclass(frozen=True)
class FreshnessDecision:
    latest_utc: dt.datetime
    now_utc: dt.datetime
    age_hours: float
    threshold_hours: int
    should_retry_refresh: bool
    ok_to_build: bool
    message: str


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def freshness_decision(
    latest_utc: dt.datetime,
    now_utc: dt.datetime | None = None,
    *,
    refresh_already_retried: bool = False,
) -> FreshnessDecision:
    """Decide whether the equity screener can build from latest 4h pricing data.

    The first gate is the user's hard reliability rule: if pricing is more than
    50h stale, re-attempt the market-data warehouse refresh once before accepting
    or refusing the build. After the retry, retain the calendar-aware build gate:
    30h on weekdays, 80h on weekends.
    """
    latest_utc = _as_utc(latest_utc)
    now_utc = _as_utc(now_utc or dt.datetime.now(dt.timezone.utc))
    now_market = now_utc.astimezone(ZoneInfo("America/New_York"))
    age_hours = (now_utc - latest_utc).total_seconds() / 3600
    threshold_hours = WEEKEND_MAX_STALE_HOURS if now_market.weekday() in (5, 6) else WEEKDAY_MAX_STALE_HOURS

    if age_hours > RETRY_STALE_HOURS and not refresh_already_retried:
        return FreshnessDecision(
            latest_utc=latest_utc,
            now_utc=now_utc,
            age_hours=age_hours,
            threshold_hours=threshold_hours,
            should_retry_refresh=True,
            ok_to_build=False,
            message=(
                f"STALE-RETRY: 4h pricing data is {age_hours:.1f}h old > {RETRY_STALE_HOURS}h; "
                "re-attempt warehouse refresh before building "
                f"(latest={latest_utc.isoformat()})"
            ),
        )

    if age_hours > threshold_hours:
        return FreshnessDecision(
            latest_utc=latest_utc,
            now_utc=now_utc,
            age_hours=age_hours,
            threshold_hours=threshold_hours,
            should_retry_refresh=False,
            ok_to_build=False,
            message=(
                f"REFUSING: 4h pricing data is {age_hours:.1f}h old > {threshold_hours}h threshold "
                f"after refresh check (latest={latest_utc.isoformat()})"
            ),
        )

    return FreshnessDecision(
        latest_utc=latest_utc,
        now_utc=now_utc,
        age_hours=age_hours,
        threshold_hours=threshold_hours,
        should_retry_refresh=False,
        ok_to_build=True,
        message=(
            f"OK: 4h pricing data is {age_hours:.1f}h old <= {threshold_hours}h threshold "
            f"(latest={latest_utc.isoformat()})"
        ),
    )


def latest_4h_pricing_timestamp(db_path: Path) -> dt.datetime:
    with duckdb.connect(str(db_path), read_only=True) as db:
        row = db.execute(
            """
            SELECT max(to_timestamp(CAST(timestamp/1000 AS BIGINT)))
            FROM technical_indicators
            WHERE timeframe='4h'
            """
        ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("no 4h RSI/pricing data in technical_indicators")
    return _as_utc(row[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Equity screener freshness guard")
    parser.add_argument("--db", type=Path, default=Path("/home/nima/market-data/market_data.duckdb"))
    parser.add_argument("--refresh-retried", action="store_true", help="Set after a warehouse refresh retry has already been attempted")
    args = parser.parse_args(argv)

    try:
        latest = latest_4h_pricing_timestamp(args.db)
    except Exception as exc:
        print(f"FATAL: {exc}")
        return 2

    decision = freshness_decision(latest, refresh_already_retried=args.refresh_retried)
    print(decision.message)
    if decision.should_retry_refresh:
        return EXIT_RETRY_REFRESH
    if not decision.ok_to_build:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
