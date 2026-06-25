#!/usr/bin/env python3
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd

from equity_screener.baskets import factor_basket_analysis, keyword_theme_analysis
from equity_screener.combined import combined_top25_opportunities
from equity_screener.polygon_overlay import enrich_latest_polygon_prices
from equity_screener.divergence import top10_factor_alignment
from equity_screener.scoring import score_candidates
from equity_screener.selection import build_diversified_top10, mark_diversified_top10
from equity_screener.serialization import record

build_dashboard = SimpleNamespace(
    combined_top25_opportunities=combined_top25_opportunities,
    build_diversified_top10=build_diversified_top10,
    factor_basket_analysis=factor_basket_analysis,
    keyword_theme_analysis=keyword_theme_analysis,
    enrich_latest_polygon_prices=enrich_latest_polygon_prices,
    mark_diversified_top10=mark_diversified_top10,
    record=record,
    score_candidates=score_candidates,
)


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
        scored = build_dashboard.mark_diversified_top10(build_dashboard.score_candidates(df))
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
    def test_top10_factor_alignment_flags_avoid_basket_divergence(self):
        top10 = pd.DataFrame({
            'ticker': ['BAD', 'GOOD'],
            'company': ['Broken Setup Inc', 'Confirmed Setup Inc'],
            'sector': ['Tech', 'Industrials'],
            'production_factor_basket': ['Broken Momentum / Avoid', 'Quality / Low-Vol Uptrend'],
            'primary_strategy': ['RSI inflection + value', 'momentum leader'],
            'opportunity_score': [88, 82],
            'rsi0': [42, 61],
            'ret_1m_pct': [-12, 4],
            'wave_stage': ['recovery attempt', 'wave 3 markup'],
        })

        out = top10_factor_alignment(top10)

        self.assertEqual(list(out['alignment_status']), ['DIVERGENCE', 'CONFIRMATION'])
        self.assertIn('conflicts with Broken Momentum / Avoid', out.loc[0, 'alignment_takeaway'])
        self.assertIn('confirms', out.loc[1, 'alignment_takeaway'])

    def test_combined_top25_prioritizes_confirming_high_composite_names(self):
        rows = []
        for i in range(24):
            rows.append({
                'ticker': f'C{i:02d}',
                'company': f'Confirmed {i}',
                'sector': f'S{i % 8}',
                'production_factor_basket': 'Quality / Low-Vol Uptrend',
                'production_factor_score': 92 - i * 0.4,
                'primary_keyword_factor': 'Cloud Software',
                'primary_keyword_factor_score': 88 - i * 0.2,
                'primary_strategy': 'momentum leader',
                'opportunity_score': 90 - i * 0.3,
                'ev_score': 86 - i * 0.2,
                'theme_reversal_score': 75,
                'rsi0': 55,
                'ret_1m_pct': 8,
                'wave_stage': 'Wave 3 / markup leader',
            })
        rows.append({
            'ticker': 'BROK',
            'company': 'Broken But High Score',
            'sector': 'S9',
            'production_factor_basket': 'Broken Momentum / Avoid',
            'production_factor_score': 5,
            'primary_keyword_factor': 'Broken Theme',
            'primary_keyword_factor_score': 20,
            'primary_strategy': 'shorted near lows / peer lag',
            'opportunity_score': 99,
            'ev_score': 99,
            'rsi0': 30,
            'ret_1m_pct': -20,
            'wave_stage': 'Wave 1 / accumulation',
        })

        out = build_dashboard.combined_top25_opportunities(pd.DataFrame(rows), limit=25, sector_cap=4)

        self.assertEqual(len(out), 25)
        self.assertIn('combined_rank_score', out.columns)
        self.assertIn('alignment_status', out.columns)
        self.assertEqual(out.iloc[0]['alignment_status'], 'CONFIRMATION')
        self.assertLess(out.index[out['ticker'].eq('BROK')][0], len(out))
        self.assertLess(out.loc[out['ticker'].eq('BROK'), 'combined_rank_score'].iloc[0], out.iloc[0]['combined_rank_score'])
        self.assertLessEqual(out.groupby('sector').size().max(), 4)

    def test_generated_dashboard_has_revamp_navigation_and_jekyll_scaffold(self):
        index = (PROJECT_DIR / 'docs' / 'index.html').read_text()
        factor = (PROJECT_DIR / 'docs' / 'factor-baskets.html').read_text()
        divergence = (PROJECT_DIR / 'docs' / 'divergence.html').read_text()
        config = (PROJECT_DIR / 'docs' / '_config.yml').read_text()
        layout = (PROJECT_DIR / 'docs' / '_layouts' / 'default.html').read_text()

        for marker in [
            'revamp-v3',
            'class="site-hero"',
            'class="rail"',
            'class="kpi-strip"',
            'Signal methodology',
            'Factor / theme map',
        ]:
            self.assertIn(marker, index)
        self.assertIn('Factor + Theme Map', factor)
        self.assertIn('href="divergence.html"', index)
        self.assertIn('id="tab-top25"', index)
        self.assertIn('Combined Top 25', index)
        self.assertIn('Top 25 combined ranking', index)
        self.assertIn('combined-rank-score', index)
        self.assertNotIn('Top 25 diversified ranked opportunities', index)
        self.assertIn('href="divergence.html"', factor)
        self.assertIn('Top-10 Divergence Monitor', divergence)
        self.assertIn('DIVERGENCE', divergence)
        self.assertIn('CONFIRMATION', divergence)
        self.assertIn('vs Broken Momentum / Avoid', divergence)
        self.assertIn('factor-alignment-card', divergence)
        self.assertIn('basket-names', factor)
        self.assertIn('Visible theme constituents', factor)
        self.assertIn('Cloud Software constituents', factor)
        self.assertIn('ticker-chip', factor)
        self.assertIn('Best opportunities within selected keyword theme', factor)
        self.assertIn('finviz.com/stock?t=', factor)
        self.assertRegex(factor, r'(FOXA|TRMB|DBX|SAIL)')
        self.assertIn('.dashboard-app .tab-content .table-wrap{display:none}', factor)
        self.assertIn('.page-shell .table-wrap{display:block;overflow:auto}', factor)
        self.assertNotIn('.footer{font-size:11px}.table-wrap{display:none}', factor)
        self.assertNotIn('.footer{font-size:11px}.table-wrap{display:none}', index)
        self.assertIn('theme: minima', config)
        self.assertIn('{{ content }}', layout)
        self.assertNotIn('{{D_DATA}}', index)
        self.assertNotIn('__DATA_PLACEHOLDER__', index)
        self.assertNotIn('(top10)', index)
        self.assertNotIn('Trader cockpit, not a spreadsheet', index)
        self.assertNotIn('Jekyll-compatible static theme surface', factor)

    def test_score_candidates_populates_defined_momentum_rs_and_wave_factors(self):
        df = pd.DataFrame({
            'sector': ['A', 'A', 'B', 'B', 'C', 'C'],
            'ticker': [f'X{i}' for i in range(6)],
            'yf_forward_pe': [10, 12, 18, 25, 8, 30],
            'yf_trailing_pe': [12, 15, 20, 28, 10, 35],
            'yf_price_to_book': [1.2, 1.5, 2.0, 3.0, 1.1, 4.0],
            'yf_peg_ratio': [1.0, 1.2, 1.5, 2.0, 0.8, 2.5],
            'dolt_value_score': [5, 6, 4, 3, 7, 2],
            'value_grade': ['C', 'B', 'C', 'D', 'A', 'D'],
            'short_pct_float': [2, 8, 12, 4, 15, 6],
            'from_52w_low_pct': [15, 35, 70, 25, 10, 120],
            'from_52w_high_pct': [-45, -20, -8, -35, -55, -2],
            'ret_1w_pct': [-2, 3, -5, 1, -8, 4],
            'ret_1m_pct': [4, 8, 16, -3, -12, 30],
            'ret_3m_pct': [8, 20, 45, -5, -20, 80],
            'ret_6m_pct': [12, 35, 70, -10, -30, 120],
            'ret_ytd_pct': [5, 15, 30, -8, -20, 60],
            'price_vs_sma20_pct': [-1, 2, 6, -3, -8, 12],
            'price_vs_sma50_pct': [0, 4, 10, -5, -15, 18],
            'price_vs_sma200_pct': [5, 15, 35, -12, -30, 60],
            'volume_vs_20d': [0.9, 1.1, 1.5, 0.7, 2.0, 1.3],
            'dollar_volume_20d_polygon': [0]*6,
            'rsi0': [42, 58, 64, 38, 30, 72],
            'rsi1': [40, 55, 60, 39, 32, 68],
            'rsi2': [41, 52, 58, 41, 35, 64],
            'rsi3': [42, 50, 55, 43, 38, 60],
            'rsi4': [43, 48, 52, 45, 40, 57],
            'rsi5': [44, 47, 50, 46, 42, 55],
            'rsi_delta_1': [2, 3, 4, -1, -2, 4],
            'prior_delta_3_avg': [-1, 2, 3, -2, -3, 4],
            'rsi_accel': [3, 1, 1, 1, 1, 0],
            'inflection_flag': [1, 0, 0, 0, 0, 0],
            'sentiment_score': [0]*6,
            'growth_grade': ['C']*6,
            'momentum_grade': ['C']*6,
            'production_factor_basket': ['x']*6,
            'production_factor_score': [50]*6,
            'production_theme': ['theme']*6,
            'primary_keyword_factor': ['kw']*6,
            'primary_keyword_factor_score': [50]*6,
            'keyword_factor_baskets': ['[]']*6,
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', pd.errors.PerformanceWarning)
            scored = build_dashboard.score_candidates(df)
        perf_warnings = [w for w in caught if issubclass(w.category, pd.errors.PerformanceWarning)]
        self.assertEqual(perf_warnings, [])
        for col in ['momentum_leader_score', 'momentum_pullback_score', 'rel_strength_pullback_score', 'inflect_breakout_score', 'wave_setup_score']:
            self.assertTrue(scored[col].notna().all(), col)
            self.assertTrue(scored[col].between(0, 100).all(), col)
            self.assertGreater(scored[col].min(), 0, col)
        self.assertIn('wave_stage', scored)
        self.assertTrue(scored['wave_stage'].notna().all())


if __name__ == '__main__':
    unittest.main()
