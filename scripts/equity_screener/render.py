import datetime as dt
import html
import json
import math

import pandas as pd

from .baskets import factor_basket_analysis, keyword_theme_analysis
from .config import DATA_DIR, DOCS_DIR, SCORE_DISPLAY
from .render_helpers import fmt_bn, fmt_money, fmt_num, render_bar, render_rank_badge, render_rsi_cell, render_score_cell, render_sparkline, ticker_link
from .selection import build_diversified_top10, cap_by_sector
from .serialization import record

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
    leaders = cap_by_sector(df[df["momentum_leader_eligible"]].sort_values("momentum_leader_score", ascending=False), "momentum_leader_score", 15, 3)
    pullbacks = cap_by_sector(df[df["mom_pullback_eligible"]].sort_values("momentum_pullback_score", ascending=False), "momentum_pullback_score", 15, 3)
    rs_pullbacks = cap_by_sector(df[df["rs_pullback_eligible"]].sort_values("rel_strength_pullback_score", ascending=False), "rel_strength_pullback_score", 25, 3)
    inflect_breakouts = cap_by_sector(df[df["inflect_breakout_eligible"]].sort_values("inflect_breakout_score", ascending=False), "inflect_breakout_score", 15, 3)
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
        "momentum_leaders": [record(r) for _, r in leaders.iterrows()],
        "momentum_pullbacks": [record(r) for _, r in pullbacks.iterrows()],
        "rel_strength_pullbacks": [record(r) for _, r in rs_pullbacks.iterrows()],
        "inflect_breakouts": [record(r) for _, r in inflect_breakouts.iterrows()],
        "master_opportunities": [record(r) for _, r in master_ev.iterrows()],
        "by_sector": [record(r) for _, r in top_sector.iterrows()],
        "llm_analysis": analyses,
    }
    (DATA_DIR / "dashboard_data.json").write_text(json.dumps(payload, indent=2, default=str))
    (DOCS_DIR / "dashboard_data.json").write_text(json.dumps(payload, indent=2, default=str))

    def row_html(r, rank_field="global_rank"):
        badge = render_rank_badge(r)
        return (
            "<tr>"
            f"<td>{badge}{int(r[rank_field]) if rank_field in r and not pd.isna(r[rank_field]) else ''}</td>"
            f"<td><strong>{ticker_link(r['ticker'])}</strong><small>{html.escape(str(r['company'])[:42])}</small></td>"
            f"<td>{html.escape(str(r['sector']))}<small>{html.escape(str(r.get('primary_strategy', '')))}</small></td>"
            f"<td>{fmt_money(r['display_close'])}<small>{html.escape(str(r.get('price_source', '')))}</small></td>"
            f"<td>{fmt_bn(r['market_cap'])}</td>"
            f"{render_score_cell(r['opportunity_score'])}"
            f"{render_rsi_cell(r)}"
            f"{render_score_cell(r.get('squeeze_laggard_score'), 'short')}"
            f"{render_score_cell(r.get('value_laggard_score'), 'value')}"
            f"{render_score_cell(r.get('momentum_leader_score'), 'lead')}"
            f"{render_score_cell(r.get('momentum_pullback_score'), 'mom')}"
            f"{render_score_cell(r.get('rel_strength_pullback_score'), 'rs')}"
            f"{render_score_cell(r.get('inflect_breakout_score'), 'brk')}"
            f"{render_score_cell(r.get('wave_setup_score'), 'wave')}"
            f"<td>{html.escape(str(r.get('wave_stage', '')))}</td>"
            f"<td>{render_sparkline(r)}</td>"
            f"<td>{fmt_num(r.get('short_pct_float'))}%</td>"
            f"<td>{fmt_num(r.get('from_52w_low_pct'))}%</td>"
            f"<td>{fmt_num(r.get('peer_lag_1m_pct'))}%</td>"
            "</tr>"
        )

    def master_row_html(r, rank_field="global_rank"):
        """Master Opportunities comparison row: all 5 sleeve scores side by side with visual bars."""
        badge = render_rank_badge(r)
        return (
            "<tr>"
            f"<td>{badge}{int(r[rank_field]) if rank_field in r and not pd.isna(r[rank_field]) else ''}</td>"
            f"<td><strong>{ticker_link(r['ticker'])}</strong><small>{html.escape(str(r['company'])[:42])}</small></td>"
            f"<td>{html.escape(str(r['sector']))}<small>{html.escape(str(r.get('primary_strategy', '')))}</small></td>"
            f"<td>{fmt_money(r['display_close'])}</td>"
            f"<td>{fmt_bn(r['market_cap'])}</td>"
            f"{render_score_cell(r['opportunity_score'])}"
            f"{render_rsi_cell(r)}"
            "<td class='master-comparison'>"
            + "".join(
                f"<div class='master-comp-row'><span class='comp-label'>{label}</span>{render_bar(r.get(score_col), cls)}</div>"
                for label, score_col, cls in SCORE_DISPLAY[1:]
            )
            + "</td>"
            f"<td>{render_sparkline(r)}</td>"
            f"<td>{fmt_num(r.get('short_pct_float'))}%</td>"
            f"<td>{fmt_num(r.get('peer_lag_1m_pct'))}%</td>"
            "</tr>"
        )

    def card_html(r, rank_field="global_rank"):
        """Mobile card layout: compact vertical card with full-width score bars."""
        badge = render_rank_badge(r)
        rank_val = int(r[rank_field]) if rank_field in r and not pd.isna(r[rank_field]) else None
        rank_str = f"<span class='mc-rank'>{badge}#{rank_val}</span>" if rank_val is not None else ""
        ticker = r['ticker']
        company = html.escape(str(r['company'])[:42])
        sector = html.escape(str(r['sector']))
        price = fmt_money(r['display_close'])
        mcap = fmt_bn(r['market_cap'])
        rsi_val = r.get('rsi0')
        rsi_str = f"{float(rsi_val):.0f}" if rsi_val is not None and not (isinstance(rsi_val, float) and math.isnan(rsi_val)) else "—"

        def _score_bar(value, score_cls=""):
            v = 0 if value is None or (isinstance(value, float) and math.isnan(value)) else max(0, min(100, float(value)))
            return f'<span class="bar {score_cls}"><i style="width:{v:.1f}%"></i></span><b>{v:.0f}</b>'

        return (
            "<div class='mobile-card'>"
            f"<div class='mc-head'>"
            f"{rank_str}"
            f"<span class='mc-ticker'>{ticker_link(ticker)}</span>"
            f"<span class='mc-company'>{company}</span>"
            f"</div>"
            f"<div class='mc-meta'><span>{sector}</span> &middot; <span>{price}</span> &middot; <span>{mcap}</span></div>"
            f"<div class='mc-scores'>"
            + "".join(
                f"<div class='mc-score-row'><span class='mc-label'>{label}</span>{_score_bar(r.get(score_col), cls)}</div>"
                for label, score_col, cls in SCORE_DISPLAY
            )
            + f"</div>"
            f"<div class='mc-details'>"
            f"<span>RSI {rsi_str}</span> &middot; "
            f"<span>Shrt {fmt_num(r.get('short_pct_float'))}%</span> &middot; "
            f"<span>Lo {fmt_num(r.get('from_52w_low_pct'))}%</span> &middot; "
            f"<span>Lag {fmt_num(r.get('peer_lag_1m_pct'))}%</span> &middot; "
            f"<span>{html.escape(str(r.get('wave_stage', '')))}</span>"
            f"</div>"
            "</div>"
        )

    top_rows = [row_html(r) for _, r in top.iterrows()]
    div_rows = []
    div_cards = []
    for display_rank, (_, r) in enumerate(diversified_top.iterrows(), 1):
        rr = r.copy()
        rr["display_rank"] = display_rank
        div_rows.append(row_html(rr, "display_rank"))
        div_cards.append(card_html(rr, "display_rank"))
    inflect_rows = [row_html(r) for _, r in inflect.iterrows()]
    inflect_cards = [card_html(r) for _, r in inflect.iterrows()]
    squeeze_rows = [row_html(r) for _, r in squeeze.iterrows()]
    squeeze_cards = [card_html(r) for _, r in squeeze.iterrows()]
    laggard_rows = [row_html(r) for _, r in laggards.iterrows()]
    laggard_cards = [card_html(r) for _, r in laggards.iterrows()]
    leader_rows = [row_html(r) for _, r in leaders.iterrows()]
    leader_cards = [card_html(r) for _, r in leaders.iterrows()]
    pullback_rows = [row_html(r) for _, r in pullbacks.iterrows()]
    pullback_cards = [card_html(r) for _, r in pullbacks.iterrows()]
    inflect_breakout_rows = [row_html(r) for _, r in inflect_breakouts.iterrows()]
    inflect_breakout_cards = [card_html(r) for _, r in inflect_breakouts.iterrows()]
    rs_pullback_rows = [row_html(r) for _, r in rs_pullbacks.iterrows()]
    rs_pullback_cards = [card_html(r) for _, r in rs_pullbacks.iterrows()]
    master_rows = [master_row_html(r) for _, r in master_ev.iterrows()]
    master_cards = [card_html(r) for _, r in master_ev.iterrows()]

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

    header = "<table><thead><tr><th>#</th><th>Ticker</th><th>Sector / Sleeve</th><th>4h Px</th><th>MCap</th><th>Opp</th><th>RSI</th><th>Short/Lows</th><th>Value/Lag</th><th>Leader</th><th>Momo Pb</th><th>Rel Str</th><th>RSI Brk</th><th>Wave</th><th>Stage</th><th>RSI 6-Period</th><th>Short%</th><th>From Low</th><th>Peer Lag 1M</th></tr></thead><tbody>"

    master_header = "<table><thead><tr><th>#</th><th>Ticker</th><th>Sector / Sleeve</th><th>4h Px</th><th>MCap</th><th>Opp</th><th>RSI</th><th>All Sleeve Scores Compared</th><th>RSI 6-Period</th><th>Short%</th><th>Peer Lag 1M</th></tr></thead><tbody>"

    factor_baskets, factor_opps = factor_basket_analysis(df)
    factor_rows = []
    selected_basket = "none"
    if not factor_baskets.empty:
        selected_basket = str(factor_baskets.iloc[0]["basket_name"])
        for _, b in factor_baskets.iterrows():
            cls = " class='selected'" if str(b["basket_name"]) == selected_basket else ""
            factor_rows.append(
                f"<tr{cls}><td>{int(b['rank'])}</td><td><strong>{html.escape(str(b['basket_name']))}</strong><small>{int(b['ticker_count'])} names under ${price_filter:.0f}</small><small class='basket-names'>{html.escape(str(b.get('top_names', '')))}</small></td>"
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
                f"<tr{cls}><td>{int(b['rank'])}</td><td><strong>{html.escape(str(b['theme_name']))}</strong><small>{int(b['ticker_count'])} names under ${price_filter:.0f}</small><small class='basket-names'>{html.escape(str(b.get('top_names', '')))}</small></td>"
                f"<td>{render_bar(b['theme_reversal_score'])}</td><td>{fmt_num(b['avg_keyword_score'])}</td><td>{fmt_num(b['avg_ret_1w_pct'])}%</td><td>{fmt_num(b['avg_ret_1m_pct'])}%</td><td>{fmt_num(b['avg_ret_3m_pct'])}%</td>"
                f"<td>{fmt_num(b['avg_rsi'])}</td><td>{fmt_num(b['avg_rsi_delta_1'])}</td><td>{fmt_num(b['avg_rsi_accel'])}</td><td>{int(b['inflection_count'])}</td></tr>"
            )
    theme_opp_rows = []
    if not theme_opps.empty:
        for display_rank, (_, r) in enumerate(theme_opps.iterrows(), 1):
            rr = r.copy()
            rr["display_rank"] = display_rank
            theme_opp_rows.append(row_html(rr, "display_rank"))

    def constituent_chips(rows: pd.DataFrame, limit: int = 12) -> str:
        if rows.empty:
            return "<span class='empty-chip'>No tickers available</span>"
        chips = []
        sort_cols = [c for c in ["opportunity_score", "ticker"] if c in rows.columns]
        sorted_rows = rows.sort_values(sort_cols, ascending=[False, True][:len(sort_cols)]) if sort_cols else rows
        for _, r in sorted_rows.head(limit).iterrows():
            ticker = str(r.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            company = html.escape(str(r.get("company", "")).strip()[:46])
            sector = html.escape(str(r.get("sector", "")).strip())
            score = fmt_num(r.get("opportunity_score"))
            chips.append(
                "<span class='ticker-chip'>"
                f"<b>{ticker_link(ticker)}</b>"
                f"<em>{company}</em>"
                f"<small>{sector} · Opp {score}</small>"
                "</span>"
            )
        return "".join(chips) or "<span class='empty-chip'>No tickers available</span>"

    def basket_cards(baskets: pd.DataFrame, group_col: str, name_col: str, score_col: str, label: str, limit: int = 8) -> str:
        if baskets.empty:
            return "<div class='constituent-empty'>No baskets available.</div>"
        cards = []
        for _, b in baskets.head(limit).iterrows():
            name = str(b.get(name_col, ""))
            group_rows = df[df[group_col].astype(str) == name]
            cards.append(
                "<article class='constituent-card'>"
                f"<div class='constituent-head'><span>{html.escape(label)}</span><strong>{html.escape(name)}</strong><b>{fmt_num(b.get(score_col))}</b></div>"
                f"<p>{int(b.get('ticker_count', 0))} names under ${price_filter:.0f}</p>"
                f"<div class='ticker-chip-grid'>{constituent_chips(group_rows, 10)}</div>"
                "</article>"
            )
        return "".join(cards)

    factor_cards_html = basket_cards(factor_baskets, "production_factor_basket", "basket_name", "factor_reversal_score", "Factor")
    theme_cards_html = basket_cards(theme_baskets, "primary_keyword_factor", "theme_name", "theme_reversal_score", "Theme", 12)
    selected_theme_chips = constituent_chips(theme_opps, 20)

    latest_price_ok = sum(
        1
        for rows in [diversified_top, top, inflect, squeeze, laggards, leaders, pullbacks, rs_pullbacks, inflect_breakouts, master_ev]
        for _, r in rows.iterrows()
        if str(r.get("latest_polygon_price_status", "")) == "OK"
    )
    avg_opp = float(df["opportunity_score"].mean()) if not df.empty else float("nan")
    avg_rsi = float(df["rsi0"].mean()) if "rsi0" in df and df["rsi0"].notna().any() else float("nan")
    kpis = [
        ("Universe", f"{len(df):,}", f"{df['sector'].nunique() if not df.empty else 0} sectors"),
        ("Live Px", f"{latest_price_ok:,}", "Polygon final-row overlays"),
        ("Avg Opp", fmt_num(avg_opp), "full filtered universe"),
        ("Avg RSI", fmt_num(avg_rsi), "latest 4h oscillator"),
        ("Research", f"{len(analyses):,}", "web-sourced top names"),
    ]
    kpi_html = "".join(
        f"<article class='kpi-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><em>{html.escape(detail)}</em></article>"
        for label, value, detail in kpis
    )

    def stage_intro(kicker: str, title: str, text: str, count: int) -> str:
        return (
            "<div class='stage-intro'>"
            f"<span class='kicker'>{html.escape(kicker)}</span>"
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(text)}</p>"
            f"<b>{int(count)} rows</b>"
            "</div>"
        )

    def count_badge(label: str, count: int) -> str:
        return f"<span>{html.escape(label)}</span><b>{int(count)}</b>"

    css = """
:root{--bg:#050607;--bg2:#090d10;--panel:#0d1317;--panel2:#111a20;--panel3:#17242c;--text:#f4efe1;--muted:#a9a294;--faint:#6f756f;--line:rgba(244,239,225,.12);--amber:#f0b83e;--green:#42d68c;--red:#ff6b6b;--purple:#c69cff;--cyan:#55d8ff;--blue:#7aa7ff;--ink:#050607;--shadow:0 20px 80px rgba(0,0,0,.42)}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;min-height:100dvh;background:radial-gradient(circle at 12% -10%,rgba(85,216,255,.18),transparent 30%),radial-gradient(circle at 90% 0,rgba(240,184,62,.15),transparent 28%),linear-gradient(135deg,var(--bg),var(--bg2));color:var(--text);font-family:ui-monospace,'SFMono-Regular','JetBrains Mono',Menlo,Consolas,monospace;font-size:13px;line-height:1.45;-webkit-font-smoothing:antialiased}a{color:inherit}body:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(to bottom,rgba(0,0,0,.9),transparent 85%)}.site-hero{position:sticky;top:0;z-index:20;border-bottom:1px solid var(--line);background:rgba(5,6,7,.90);backdrop-filter:blur(18px);box-shadow:0 10px 40px rgba(0,0,0,.22)}.hero-inner{max-width:1760px;margin:0 auto;padding:18px 24px;display:grid;grid-template-columns:1.2fr auto;gap:16px;align-items:center}.brand-lockup{display:flex;gap:14px;align-items:center}.brand-mark{width:48px;height:48px;border:1px solid rgba(240,184,62,.5);border-radius:15px;display:grid;place-items:center;background:linear-gradient(145deg,rgba(240,184,62,.22),rgba(85,216,255,.08));box-shadow:inset 0 0 24px rgba(240,184,62,.12)}.brand-copy h1{margin:0;font-size:22px;letter-spacing:.09em;text-transform:uppercase;color:var(--amber)}.brand-copy p{margin:4px 0 0;color:var(--muted)}.hero-actions{display:flex;gap:10px;align-items:center;justify-content:flex-end;flex-wrap:wrap}.status-pill,.hero-actions a{min-height:36px;display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:8px 12px;background:rgba(255,255,255,.035);color:var(--muted);text-decoration:none}.status-pill.live{color:var(--green);border-color:rgba(66,214,140,.35)}.hero-actions a.active{color:var(--ink);background:var(--amber);border-color:var(--amber);font-weight:800}.dashboard-app{max-width:1760px;margin:0 auto;padding:18px 24px 36px}.tab-switch{position:absolute;left:-9999px}.workspace{display:grid;grid-template-columns:250px minmax(0,1fr);gap:18px}.rail{position:sticky;top:94px;align-self:start;border:1px solid var(--line);border-radius:24px;background:linear-gradient(180deg,rgba(17,26,32,.95),rgba(8,12,15,.92));box-shadow:var(--shadow);padding:14px}.rail-title{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.12em;margin:2px 4px 12px}.rail-title b{color:var(--amber)}.rail label,.rail a{width:100%;min-height:44px;display:flex;justify-content:space-between;align-items:center;padding:10px 12px;margin:4px 0;border-radius:14px;border:1px solid transparent;color:var(--muted);text-decoration:none;cursor:pointer;touch-action:manipulation}.rail label:hover,.rail a:hover{background:rgba(255,255,255,.045);color:var(--text);border-color:var(--line)}.rail label b{color:var(--faint);font-weight:600}.rail a.factor-link{margin-top:10px;border-color:rgba(85,216,255,.22);color:var(--cyan)}#tab-opps:checked~.workspace .opps-tab,#tab-rsi:checked~.workspace .rsi-tab,#tab-sqz:checked~.workspace .sqz-tab,#tab-val:checked~.workspace .val-tab,#tab-lead:checked~.workspace .lead-tab,#tab-mom:checked~.workspace .mom-tab,#tab-brk:checked~.workspace .brk-tab,#tab-rspb:checked~.workspace .rspb-tab,#tab-master:checked~.workspace .master-tab,#tab-sector:checked~.workspace .sector-tab{background:linear-gradient(135deg,var(--amber),#ffd978);color:var(--ink);border-color:var(--amber);box-shadow:0 10px 34px rgba(240,184,62,.22)}.main-stage{min-width:0}.kpi-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:14px}.kpi-card{border:1px solid var(--line);border-radius:20px;background:linear-gradient(160deg,rgba(23,36,44,.92),rgba(8,12,15,.9));padding:14px 16px;min-height:110px;position:relative;overflow:hidden}.kpi-card:after{content:"";position:absolute;right:-28px;top:-34px;width:95px;height:95px;border-radius:50%;background:rgba(240,184,62,.08)}.kpi-card span{display:block;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;font-size:10px}.kpi-card strong{display:block;margin:8px 0 2px;font-size:27px;color:var(--text);letter-spacing:-.04em}.kpi-card em{font-style:normal;color:var(--faint);font-size:11px}.method-panel{border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(240,184,62,.10),rgba(85,216,255,.07),rgba(255,255,255,.02));padding:18px;margin-bottom:14px;display:grid;grid-template-columns:1.2fr .8fr;gap:16px}.method-panel h2{margin:0 0 8px;color:var(--amber);font-size:16px;text-transform:uppercase;letter-spacing:.08em}.method-panel p{margin:0;color:var(--muted)}.method-facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.method-facts span{border:1px solid var(--line);border-radius:14px;padding:10px;color:var(--muted);background:rgba(0,0,0,.16)}.method-facts b{display:block;color:var(--text);font-size:15px}.tab-nav{position:sticky;top:86px;z-index:15;display:flex;gap:8px;overflow-x:auto;padding:10px;margin:0 0 16px;border:1px solid var(--line);border-radius:22px;background:rgba(5,6,7,.82);backdrop-filter:blur(16px);-webkit-overflow-scrolling:touch}.tab-nav label{min-height:44px;white-space:nowrap;display:inline-flex;align-items:center;gap:9px;padding:10px 13px;border-radius:15px;border:1px solid var(--line);background:rgba(255,255,255,.035);color:var(--muted);cursor:pointer;touch-action:manipulation}.tab-nav label:hover{color:var(--text);background:rgba(255,255,255,.07)}.tab-nav label b{font-size:11px;color:var(--faint)}.tab-content{display:none;animation:rise .24s ease-out}#tab-opps:checked~.workspace #c-opps,#tab-rsi:checked~.workspace #c-rsi,#tab-sqz:checked~.workspace #c-sqz,#tab-val:checked~.workspace #c-val,#tab-lead:checked~.workspace #c-lead,#tab-mom:checked~.workspace #c-mom,#tab-brk:checked~.workspace #c-brk,#tab-rspb:checked~.workspace #c-rspb,#tab-master:checked~.workspace #c-master,#tab-sector:checked~.workspace #c-sector{display:block}@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}.stage-intro{border:1px solid var(--line);border-radius:24px;background:linear-gradient(145deg,rgba(17,26,32,.96),rgba(8,12,15,.94));padding:18px;margin:0 0 14px;display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end}.stage-intro .kicker{grid-column:1/-1;color:var(--cyan);text-transform:uppercase;letter-spacing:.16em;font-size:10px}.stage-intro h2{margin:0;color:var(--text);font-size:22px;letter-spacing:-.03em}.stage-intro p{margin:0;color:var(--muted);max-width:900px}.stage-intro b{border:1px solid rgba(240,184,62,.35);border-radius:999px;padding:8px 12px;color:var(--amber);white-space:nowrap}.signal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:0 0 18px}.mobile-card{border:1px solid var(--line);border-radius:22px;background:linear-gradient(160deg,rgba(17,26,32,.95),rgba(8,12,15,.92));box-shadow:0 16px 50px rgba(0,0,0,.25);padding:14px;position:relative;overflow:hidden}.mobile-card:before{content:"";position:absolute;inset:0 0 auto;height:3px;background:linear-gradient(90deg,var(--amber),var(--cyan),var(--green))}.mc-head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:8px}.mc-rank{color:var(--amber);font-weight:900}.mc-ticker a,.mc-ticker{font-size:18px;font-weight:900;color:var(--text);text-decoration:none}.mc-company{color:var(--muted);font-size:11px}.mc-meta{color:var(--muted);font-size:11px;margin-bottom:10px}.mc-scores{display:grid;gap:6px}.mc-score-row{display:grid;grid-template-columns:46px minmax(0,1fr) 32px;align-items:center;gap:8px}.mc-label{color:var(--muted);font-size:10px;text-align:right}.bar{height:12px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;width:0;border-radius:inherit;background:linear-gradient(90deg,var(--amber),var(--green))}.bar.short i{background:linear-gradient(90deg,var(--cyan),var(--blue))}.bar.value i{background:linear-gradient(90deg,var(--purple),var(--amber))}.bar.lead i{background:linear-gradient(90deg,var(--blue),var(--cyan))}.bar.mom i{background:linear-gradient(90deg,var(--red),var(--amber))}.bar.rs i{background:linear-gradient(90deg,#ffa500,var(--green))}.bar.brk i{background:linear-gradient(90deg,var(--cyan),var(--green))}.mobile-card .bar+b{color:var(--text);font-size:11px;text-align:right}.mc-details{border-top:1px solid var(--line);margin-top:10px;padding-top:8px;color:var(--faint);font-size:11px}.data-panel{border:1px solid var(--line);border-radius:24px;background:rgba(8,12,15,.9);overflow:hidden;margin-bottom:18px;box-shadow:var(--shadow)}.panel-title{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:14px 16px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.03)}.panel-title h3{margin:0;color:var(--amber);font-size:14px;text-transform:uppercase;letter-spacing:.08em}.panel-title span{color:var(--muted);font-size:11px}.table-wrap{overflow:auto;-webkit-overflow-scrolling:touch}table{width:100%;border-collapse:separate;border-spacing:0;background:transparent}th,td{border-bottom:1px solid var(--line);padding:10px 9px;text-align:left;vertical-align:top}th{position:sticky;top:0;z-index:2;background:#0c1115;color:var(--amber);font-size:11px;text-transform:uppercase;letter-spacing:.08em}td{color:var(--text)}td small{display:block;color:var(--muted);font-size:11px;margin-top:3px}tr:hover td{background:rgba(255,255,255,.025)}.score-td{min-width:74px}.score-bar{height:18px;min-width:58px;background:rgba(255,255,255,.07);border-radius:99px;position:relative;overflow:hidden}.score-bar .bar-fill{position:absolute;inset:0 auto 0 0;width:var(--bar-pct);background:linear-gradient(90deg,var(--bar-color),rgba(255,255,255,.18));border-radius:inherit}.score-bar .bar-score{position:absolute;inset:0;display:grid;place-items:center;font-weight:900;color:var(--text);font-size:11px;text-shadow:0 1px 2px #000}.rsi-td .rsi-val{display:block;font-weight:900;color:var(--text)}.sparkline{display:flex;gap:2px;align-items:center}.spark-dot{font-size:11px}.spark-arrow{font-size:8px;color:var(--faint)}.spark-label{font-size:9px;color:var(--muted);margin-left:4px}.rank-badge{margin-right:4px}.master-comparison{min-width:240px}.master-comp-row{display:grid;grid-template-columns:58px 1fr;gap:6px;align-items:center;margin:3px 0}.comp-label{color:var(--muted);font-size:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.card,.sector{border:1px solid var(--line);border-radius:22px;background:linear-gradient(160deg,rgba(17,26,32,.95),rgba(8,12,15,.92));padding:14px}.card-head{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;margin-bottom:9px}.card-head span{color:var(--amber);font-size:16px;font-weight:900}.card-head em{font-style:normal;color:var(--muted)}.metrics{color:var(--green);margin-bottom:8px}.sources{border:1px solid var(--line);border-radius:14px;background:rgba(0,0,0,.20);padding:9px;margin:9px 0;color:var(--muted)}.sources b{color:var(--cyan)}.sources ol{margin:6px 0 0 18px;padding:0}.sources a{color:var(--amber);text-decoration:none}.sources.missing{border-color:rgba(255,107,107,.25)}.card p{line-height:1.58;margin:0;color:#ddd5c4}.sector h3{margin-top:0}.sector ol{margin:0;padding-left:20px}.sector li{margin:9px 0;color:var(--muted)}.pill{display:inline-flex;align-items:center;min-height:26px;border:1px solid rgba(240,184,62,.4);border-radius:999px;padding:4px 9px;color:var(--amber);background:rgba(240,184,62,.08);font-weight:800}.note{border:1px solid var(--line);border-radius:18px;padding:14px;background:rgba(255,255,255,.035);color:var(--muted);line-height:1.55;margin:0 0 14px}.footer{margin:24px 0 0;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:12px}.page-shell{max-width:1760px;margin:0 auto;padding:18px 24px 36px}.page-panel{border:1px solid var(--line);border-radius:24px;background:rgba(8,12,15,.9);padding:16px;margin-bottom:16px}.selected td{background:rgba(240,184,62,.08)!important}.sr-only{position:absolute;left:-10000px;width:1px;height:1px;overflow:hidden}@media(max-width:1100px){.workspace{grid-template-columns:1fr}.rail{position:relative;top:auto;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.rail-title{grid-column:1/-1}.kpi-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.method-panel{grid-template-columns:1fr}.tab-nav{top:78px}.hero-inner{grid-template-columns:1fr}.hero-actions{justify-content:flex-start}}@media(max-width:768px){.hero-inner{padding:12px 14px}.dashboard-app,.page-shell{padding:12px}.brand-mark{width:40px;height:40px}.brand-copy h1{font-size:16px}.brand-copy p{font-size:11px}.hero-actions{gap:6px}.status-pill,.hero-actions a{font-size:11px;min-height:34px;padding:7px 10px}.rail{display:none}.kpi-strip{grid-template-columns:1fr 1fr;gap:8px}.kpi-card{min-height:92px;padding:12px}.kpi-card strong{font-size:22px}.method-facts{grid-template-columns:1fr}.tab-nav{top:65px;border-radius:16px;margin-left:-4px;margin-right:-4px}.tab-nav label{font-size:11px;padding:8px 10px}.stage-intro{grid-template-columns:1fr;padding:14px}.stage-intro h2{font-size:18px}.signal-grid{grid-template-columns:1fr}.tab-content table,.tab-content thead,.tab-content tbody{display:none!important}.data-panel{padding:0}.panel-title{padding:12px}.grid{grid-template-columns:1fr}.mobile-card{padding:12px}.mc-score-row{grid-template-columns:44px minmax(0,1fr) 30px}.footer{font-size:11px}.table-wrap{display:none}}@supports (-webkit-touch-callout:none){body{min-height:-webkit-fill-available}.tab-nav,.table-wrap{-webkit-overflow-scrolling:touch}}
"""
    theme_css = """
.constituent-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin:14px 0 20px}.constituent-card{border:1px solid rgba(240,184,62,.18);border-radius:18px;background:linear-gradient(145deg,rgba(240,184,62,.08),rgba(85,216,255,.035));padding:16px;box-shadow:0 12px 40px rgba(0,0,0,.22)}.constituent-head{display:grid;grid-template-columns:1fr auto;gap:6px 10px;align-items:start}.constituent-head span{grid-column:1/-1;color:var(--amber);font-size:10px;text-transform:uppercase;letter-spacing:.14em}.constituent-head strong{font-size:16px;color:var(--text)}.constituent-head b{min-width:42px;text-align:center;border-radius:999px;padding:3px 8px;background:rgba(240,184,62,.2);color:var(--amber)}.constituent-card p{margin:6px 0 12px;color:var(--muted)}.ticker-chip-grid{display:flex;flex-wrap:wrap;gap:8px}.ticker-chip{display:grid;gap:2px;min-width:156px;max-width:220px;padding:9px 10px;border:1px solid rgba(244,239,225,.12);border-radius:12px;background:rgba(5,6,7,.45)}.ticker-chip b a{color:var(--text);font-size:14px;text-decoration:underline;text-decoration-color:rgba(85,216,255,.45)}.ticker-chip em{font-style:normal;color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ticker-chip small{display:block;color:var(--faint);font-size:10px}.selected-theme-strip{border:1px solid rgba(85,216,255,.24);border-radius:18px;background:rgba(85,216,255,.06);padding:16px;margin:0 0 18px}.selected-theme-strip h3{margin:0 0 4px;color:var(--cyan);text-transform:uppercase;letter-spacing:.1em}.selected-theme-strip p{margin:0 0 12px;color:var(--muted)}.empty-chip,.constituent-empty{color:var(--muted);padding:12px;border:1px dashed var(--line);border-radius:12px}@media(max-width:768px){.constituent-grid{grid-template-columns:1fr}.ticker-chip{min-width:100%;}.table-wrap{overflow-x:auto}}
"""
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Equity Screener</title><style>{css}</style></head>
<body class="revamp-v3"><header class="site-hero"><div class="hero-inner"><div class="brand-lockup"><div class="brand-mark">◈</div><div class="brand-copy"><h1>Equity Screener</h1><p>Production multi-sleeve screen · generated {html.escape(now_et.strftime('%Y-%m-%d %H:%M %Z'))}</p></div></div><div class="hero-actions"><span class="status-pill live">● Live</span><span class="status-pill">latest 4h {html.escape(latest_ts)}</span><span class="status-pill">price &lt; ${price_filter:.0f}</span><a class="active" href="index.html">Cockpit</a><a href="factor-baskets.html">Themes</a></div></div></header>
<main class="dashboard-app" id="top"><input class="tab-switch" type="radio" name="tab" id="tab-opps" checked><input class="tab-switch" type="radio" name="tab" id="tab-rsi"><input class="tab-switch" type="radio" name="tab" id="tab-sqz"><input class="tab-switch" type="radio" name="tab" id="tab-val"><input class="tab-switch" type="radio" name="tab" id="tab-lead"><input class="tab-switch" type="radio" name="tab" id="tab-mom"><input class="tab-switch" type="radio" name="tab" id="tab-brk"><input class="tab-switch" type="radio" name="tab" id="tab-rspb"><input class="tab-switch" type="radio" name="tab" id="tab-master"><input class="tab-switch" type="radio" name="tab" id="tab-sector">
<div class="workspace"><aside class="rail"><div class="rail-title"><span>Navigation</span><b>Desk</b></div><label class="opps-tab" for="tab-opps">{count_badge('Command center', len(div_rows))}</label><label class="rsi-tab" for="tab-rsi">{count_badge('RSI Inflections', len(inflect_rows))}</label><label class="sqz-tab" for="tab-sqz">{count_badge('Squeeze Laggards', len(squeeze_rows))}</label><label class="val-tab" for="tab-val">{count_badge('Value Laggards', len(laggard_rows))}</label><label class="lead-tab" for="tab-lead">{count_badge('Momentum Leaders', len(leader_rows))}</label><label class="mom-tab" for="tab-mom">{count_badge('Momentum Pullbacks', len(pullback_rows))}</label><label class="brk-tab" for="tab-brk">{count_badge('RSI Breakout', len(inflect_breakout_rows))}</label><label class="rspb-tab" for="tab-rspb">{count_badge('RS Pullbacks', len(rs_pullback_rows))}</label><label class="master-tab" for="tab-master">{count_badge('EV Master', len(master_rows))}</label><label class="sector-tab" for="tab-sector">{count_badge('By Sector', len(top_sector))}</label><a class="factor-link" href="factor-baskets.html">Factor / theme map →</a></aside>
<section class="main-stage"><section class="kpi-strip">{kpi_html}</section><section class="method-panel"><div><h2>Signal methodology</h2><p>Scores are deterministic 0-100 composites from local Polygon/DuckDB price, volume, RSI, valuation, sector, and factor data. Every defined factor is populated for every stock; eligibility flags only choose which names appear in each sleeve.</p></div><div class="method-facts"><span><b>Known</b>Prices, technicals, sectors, and factor fields from local Polygon/DuckDB.</span><span><b>Estimated</b>Composite scores from normalized warehouse fields.</span><span><b>Unknown</b>Unextracted catalysts or risks need manual source review.</span><span><b>Constraint</b>Top ten capped at max 3 names per sector.</span></div></section>
<nav class="tab-nav" aria-label="Opportunity sections"><label class="opps-tab" for="tab-opps">Command <b>{len(div_rows)}</b></label><label class="rsi-tab" for="tab-rsi">RSI <b>{len(inflect_rows)}</b></label><label class="sqz-tab" for="tab-sqz">Squeeze <b>{len(squeeze_rows)}</b></label><label class="val-tab" for="tab-val">Value <b>{len(laggard_rows)}</b></label><label class="lead-tab" for="tab-lead">Leaders <b>{len(leader_rows)}</b></label><label class="mom-tab" for="tab-mom">Momentum <b>{len(pullback_rows)}</b></label><label class="brk-tab" for="tab-brk">Breakout <b>{len(inflect_breakout_rows)}</b></label><label class="rspb-tab" for="tab-rspb">RS Pullback <b>{len(rs_pullback_rows)}</b></label><label class="master-tab" for="tab-master">EV Master <b>{len(master_rows)}</b></label><label class="sector-tab" for="tab-sector">Sectors <b>{len(top_sector)}</b></label></nav>
<div class="tab-content" id="c-opps">{stage_intro('Command center','Multi-sleeve top 10 opportunities','The fastest read: diversified top names across all sleeves, live price overlay, score stack, and web-sourced qualitative checks.', len(div_rows))}<div class="signal-grid">{''.join(div_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Top 10 opportunity table</h3><span>desktop audit trail</span></div><div class="table-wrap">{header}{''.join(div_rows)}</tbody></table></div></div><div class="data-panel"><div class="panel-title"><h3>Web-sourced deep dive</h3><span>top diversified names</span></div><div class="grid">{''.join(analysis_cards)}</div></div><div class="data-panel"><div class="panel-title"><h3>Top 25 diversified ranked opportunities</h3><span>sector capped</span></div><div class="table-wrap">{header}{''.join(top_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-rsi">{stage_intro('Inflection','RSI inflection + value','Mean-reversion/value names where 4h RSI is turning upward from washed-out or improving setups.', len(inflect_rows))}<div class="signal-grid">{''.join(inflect_cards)}</div><div class="data-panel"><div class="panel-title"><h3>RSI inflection sleeve</h3><span>ranked by RSI/value score</span></div><div class="table-wrap">{header}{''.join(inflect_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-sqz">{stage_intro('Pressure','Shorted near lows / peer lag','Short-heavy names near lows where peer lag and volume make the squeeze asymmetry visible.', len(squeeze_rows))}<div class="signal-grid">{''.join(squeeze_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Shorted near lows / peer lag sleeve</h3><span>ranked by squeeze laggard score</span></div><div class="table-wrap">{header}{''.join(squeeze_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-val">{stage_intro('Reversion','Cheap peer laggard sleeve','Cheap names that have lagged their sector/peers and may offer catch-up convexity.', len(laggard_rows))}<div class="signal-grid">{''.join(laggard_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Cheap peer laggard sleeve</h3><span>ranked by value lag score</span></div><div class="table-wrap">{header}{''.join(laggard_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-lead">{stage_intro('Markup','Momentum leaders / wave-3 candidates','Strong 3-6 month relative performers with trend alignment, constructive RSI, volume confirmation, and controlled distribution risk.', len(leader_rows))}<div class="signal-grid">{''.join(leader_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Momentum leader sleeve</h3><span>ranked by leader / wave-3 score</span></div><div class="table-wrap">{header}{''.join(leader_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-mom">{stage_intro('Continuation','Momentum pullback sleeve','Strong 6-month relative winners that pulled back recently and are coiling near moving-average support.', len(pullback_rows))}<div class="signal-grid">{''.join(pullback_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Momentum pullback sleeve</h3><span>continuation setup</span></div><div class="table-wrap">{header}{''.join(pullback_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-brk">{stage_intro('Expansion','RSI breakout / inflection sleeve','Stocks where RSI is turning up from the 40-60 mid-range with volume expansion and sector tailwind.', len(inflect_breakout_rows))}<div class="signal-grid">{''.join(inflect_breakout_cards)}</div><div class="data-panel"><div class="panel-title"><h3>RSI breakout / inflection sleeve</h3><span>early breakout setup</span></div><div class="table-wrap">{header}{''.join(inflect_breakout_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-rspb">{stage_intro('Reset','Relative strength pullback sleeve','Winning-sector stocks with strong 3-month momentum that sold off this week into support/demand zones.', len(rs_pullback_rows))}<p class="note"><span class="pill">RS Pullback</span> Gate: sector 1-month return &gt; -2%, stock near sector 3-month momentum, 1-week pullback &lt; -1%, and not too extended from the 52-week low (&lt;40%). Scored by sector strength, relative strength, pullback depth, SMA proximity, and RSI reset.</p><div class="signal-grid">{''.join(rs_pullback_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Relative strength pullback sleeve</h3><span>reset setup</span></div><div class="table-wrap">{header}{''.join(rs_pullback_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-master">{stage_intro('Expected value','Master opportunities — high expected value','A combined view of top sleeve signal, cross-sleeve agreement, asymmetric R:R, and factor alignment.', len(master_rows))}<p class="note"><span class="pill">EV Formula</span> 35% top sleeve signal + 25% cross-sleeve agreement + 25% asymmetric R:R + 15% factor alignment. Only stocks scoring ≥60 across all dimensions qualify. Sorted by EV score, capped at 3 per sector.</p><div class="signal-grid">{''.join(master_cards)}</div><div class="data-panel"><div class="panel-title"><h3>Master opportunities comparison</h3><span>all six sleeves side by side</span></div><div class="table-wrap">{master_header}{''.join(master_rows)}</tbody></table></div></div></div>
<div class="tab-content" id="c-sector">{stage_intro('Map','Top by sector','Sector-by-sector access for portfolio construction and avoiding concentration blind spots.', len(top_sector))}<div class="grid">{''.join(sector_sections)}</div></div><div class="footer">Known: numeric data from local Polygon/DuckDB warehouse; cited qualitative commentary from extracted web sources. Estimated: composite scores from normalized warehouse fields. Unknown: catalysts or risks not present in extracted sources or warehouse fields.</div></section></div></main></body></html>"""
    factor_header = "<table><thead><tr><th>#</th><th>Factor basket</th><th>Reversal</th><th>1W</th><th>1M</th><th>3M</th><th>RSI</th><th>RSI Δ1</th><th>RSI Accel</th><th>Inflect Names</th></tr></thead><tbody>"
    theme_header = "<table><thead><tr><th>#</th><th>Keyword / theme basket</th><th>Reversal</th><th>Theme Score</th><th>1W</th><th>1M</th><th>3M</th><th>RSI</th><th>RSI Δ1</th><th>RSI Accel</th><th>Inflect Names</th></tr></thead><tbody>"
    factor_content = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Factor Basket Inflections</title><style>{css}{theme_css}</style></head>
<body class="revamp-v3"><header class="site-hero"><div class="hero-inner"><div class="brand-lockup"><div class="brand-mark">◎</div><div class="brand-copy"><h1>Factor + Theme Map</h1><p>Lagging baskets, keyword themes, and under-$75 opportunity drilldowns</p></div></div><div class="hero-actions"><span class="status-pill live">● Live</span><span class="status-pill">generated {html.escape(now_et.strftime('%Y-%m-%d %H:%M %Z'))}</span><span class="status-pill">price &lt; ${price_filter:.0f}</span><a href="index.html">Cockpit</a><a class="active" href="factor-baskets.html">Themes</a></div></div></header>
<main class="page-shell"><section class="kpi-strip">{kpi_html}</section><section class="method-panel"><div><h2>Theme and factor breadth</h2><p>Production factor baskets and Polygon keyword themes show where quantitative breadth is lagging, inflecting, or leading. Drilldowns use the same populated sleeve scores as the main screen.</p></div><div class="method-facts"><span><b>Selected factor</b>{html.escape(selected_basket)}</span><span><b>Selected theme</b>{html.escape(selected_theme)}</span><span><b>Known</b>Warehouse factor and keyword fields.</span><span><b>Estimated</b>Composite reversal scores.</span></div></section><div class="page-panel"><div class="panel-title"><h3>Visible factor constituents</h3><span>tickers and company names</span></div><div class="constituent-grid">{factor_cards_html}</div></div><div class="page-panel"><div class="panel-title"><h3>Production factor basket score + momentum analysis</h3><span>basket reversal model</span></div><div class="table-wrap">{factor_header}{''.join(factor_rows)}</tbody></table></div></div><div class="page-panel"><div class="panel-title"><h3>Best opportunities within selected factor: {html.escape(selected_basket)}</h3><span>under ${price_filter:.0f}</span></div><div class="table-wrap">{header}{''.join(factor_opp_rows)}</tbody></table></div></div><div class="page-panel"><div class="panel-title"><h3>Visible theme constituents</h3><span>theme tickers and company names</span></div><div class="constituent-grid">{theme_cards_html}</div></div><div class="page-panel"><div class="panel-title"><h3>Keyword / theme basket score + momentum analysis</h3><span>beneficiary clusters</span></div><div class="table-wrap">{theme_header}{''.join(theme_rows)}</tbody></table></div></div><div class="page-panel"><div class="panel-title"><h3>Best opportunities within selected keyword theme: {html.escape(selected_theme)}</h3><span>theme drilldown</span></div><div class="selected-theme-strip"><h3>{html.escape(selected_theme)} constituents</h3><p>These are the actual tickers in the selected theme drilldown.</p><div class="ticker-chip-grid">{selected_theme_chips}</div></div><div class="table-wrap">{header}{''.join(theme_opp_rows)}</tbody></table></div></div><div class="footer">Known: production factor baskets, primary keyword factors, prices, technicals, returns, and factor scores from local Polygon/DuckDB warehouse. Estimated: reversal scores are deterministic composites of basket lag, keyword relevance, and short-term inflection.</div></main></body></html>"""
    (DOCS_DIR / "factor-baskets.html").write_text(factor_content)
    (DOCS_DIR / "index.html").write_text(content)
