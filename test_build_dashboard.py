#!/usr/bin/env python3
import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parent / 'scripts' / 'build_dashboard.py'
spec = importlib.util.spec_from_file_location('build_dashboard', MODULE_PATH)
build_dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dashboard)


class BuildDashboardContractTest(unittest.TestCase):
    def test_record_preserves_string_price_source_and_relative_strength_score(self):
        row = pd.Series({
            'global_rank': 1,
            'rank_in_sector': 1,
            'sector': 'Tech',
            'ticker': 'TEST',
            'company': 'Test Corp',
            'market_cap': 1_000_000_000,
            'four_h_close': 10.0,
            'display_close': 10.5,
            'price_source': 'daily_close_newer_than_4h',
            'latest_daily_close': 10.5,
            'rel_strength_pullback_score': 77.7,
        })
        row['latest_polygon_price'] = 10.75
        row['latest_polygon_price_source'] = 'snapshot.lastTrade.p'
        row['latest_polygon_price_timestamp'] = 1782400000000000000
        row['warehouse_display_close'] = 10.5
        out = build_dashboard.record(row)
        self.assertEqual(out['price_source'], 'daily_close_newer_than_4h')
        self.assertEqual(out['rel_strength_pullback_score'], 77.7)
        self.assertEqual(out['latest_polygon_price'], 10.75)
        self.assertEqual(out['latest_polygon_price_source'], 'snapshot.lastTrade.p')
        self.assertEqual(out['warehouse_display_close'], 10.5)

    def test_score_candidates_diversified_flag_matches_dynamic_top_builder(self):
        df = pd.DataFrame({
            'sector': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F'],
            'ticker': [f'T{i}' for i in range(12)],
            'yf_forward_pe': [10]*12,
            'yf_trailing_pe': [12]*12,
            'yf_price_to_book': [1.2]*12,
            'yf_peg_ratio': [1.0]*12,
            'dolt_value_score': [5]*12,
            'value_grade': ['C']*12,
            'short_pct_float': list(range(12)),
            'from_52w_low_pct': [20]*12,
            'from_52w_high_pct': [-20]*12,
            'ret_1w_pct': [-2]*12,
            'ret_1m_pct': [0]*12,
            'ret_3m_pct': [5]*12,
            'ret_6m_pct': list(range(12)),
            'ret_ytd_pct': [0]*12,
            'volume_vs_20d': [1.2]*12,
            'dollar_volume_20d_polygon': [0]*12,
            'rsi0': [45]*12,
            'rsi1': [43]*12,
            'rsi2': [42]*12,
            'rsi3': [41]*12,
            'rsi4': [40]*12,
            'rsi5': [39]*12,
            'rsi_delta_1': [2]*12,
            'prior_delta_3_avg': [-1]*12,
            'rsi_accel': [3]*12,
            'inflection_flag': [1]*12,
            'sentiment_score': [0]*12,
            'growth_grade': ['C']*12,
            'momentum_grade': ['C']*12,
            'production_factor_basket': ['x']*12,
            'production_factor_score': list(range(12)),
            'production_theme': ['theme']*12,
            'primary_keyword_factor': ['kw']*12,
            'primary_keyword_factor_score': list(range(12)),
            'keyword_factor_baskets': ['[]']*12,
        })
        scored = build_dashboard.score_candidates(df)
        dynamic_tickers = set(build_dashboard.build_diversified_top10(scored, 3)['ticker'])
        flagged_tickers = set(scored.loc[scored['diversified_top10'], 'ticker'])
        self.assertEqual(flagged_tickers, dynamic_tickers)
        self.assertFalse(df is scored)
    def test_enrich_latest_polygon_prices_updates_only_selected_final_candidates(self):
        class FakeClient:
            def latest_prices(self, tickers):
                self.requested = list(tickers)
                return {
                    'AAPL': {'status': 'OK', 'ticker': 'AAPL', 'price': 199.12, 'source': 'snapshot.lastTrade.p', 'timestamp': 1782400000000000000},
                    'MSFT': {'status': 'ERROR', 'ticker': 'MSFT', 'price': None, 'source': None, 'timestamp': None, 'error': 'no usable price'},
                }

        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'NVDA'],
            'display_close': [190.0, 410.0, 500.0],
            'price_source': ['4h_close', '4h_close', '4h_close'],
        })
        client = FakeClient()

        enriched = build_dashboard.enrich_latest_polygon_prices(df, ['AAPL', 'MSFT'], client=client)

        self.assertEqual(client.requested, ['AAPL', 'MSFT'])
        self.assertEqual(enriched.loc[0, 'warehouse_display_close'], 190.0)
        self.assertEqual(enriched.loc[0, 'display_close'], 199.12)
        self.assertEqual(enriched.loc[0, 'price_source'], 'polygon.snapshot.lastTrade.p')
        self.assertEqual(enriched.loc[0, 'latest_polygon_price'], 199.12)
        self.assertEqual(enriched.loc[0, 'latest_polygon_price_timestamp'], 1782400000000000000)
        self.assertEqual(enriched.loc[1, 'display_close'], 410.0)
        self.assertEqual(enriched.loc[1, 'latest_polygon_price_status'], 'ERROR')
        self.assertTrue(pd.isna(enriched.loc[2, 'latest_polygon_price']))


if __name__ == '__main__':
    unittest.main()
