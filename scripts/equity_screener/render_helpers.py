import html
import math
import urllib.parse

from .config import COLOR_RGB

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
    """Full-width gradient bar filling the cell."""
    v = 0 if value is None or (isinstance(value, float) and math.isnan(value)) else max(0, min(100, float(value)))
    color_map = {"hot": "var(--green)", "value": "var(--purple)", "short": "var(--cyan)", "lead": "var(--blue)", "mom": "var(--red)", "rs": "#ffa500", "brk": "#00ced1", "wave": "var(--green)"}
    color = color_map.get(cls, "var(--amber)")
    return f'<div class="score-bar" style="--bar-pct:{v:.0f}%;--bar-color:{color}"><span class="bar-fill"></span><span class="bar-score">{v:.0f}</span></div>'

def render_score_cell(value, cls=""):
    """Score cell with full-width bar + heatmap background."""
    v = 0 if value is None or (isinstance(value, float) and math.isnan(value)) else max(0, min(100, float(value)))
    opacity = v / 100
    # Heatmap: dark amber-to-color gradient background with opacity proportional to score
    color_rgb = COLOR_RGB.get(cls, "230,180,34")
    bg_alpha = 0.04 + opacity * 0.10  # subtle range 4%-14%
    class_attr = f' score-{html.escape(cls)}' if cls else ''
    return (
        f'<td class="score-td{class_attr}" style="background:rgba({color_rgb},{bg_alpha:.3f});--bar-pct:{v:.0f}%;--bar-color:rgb({color_rgb})">'
        f'<div class="score-bar" style="--bar-pct:{v:.0f}%;--bar-color:rgb({color_rgb})">'
        f'<span class="bar-fill"></span>'
        f'<span class="bar-score">{v:.0f}</span>'
        f'</div></td>'
    )

def render_sparkline(r):
    """Render 6 RSI dots (rsi0..rsi5) as a mini sparkline row."""
    dots = []
    for i in range(6):
        key = f"rsi{i}"
        val = r.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            dots.append('<span class="spark-dot dot-none" title="no data">○</span>')
        else:
            v = float(val)
            if v < 30:
                dots.append(f'<span class="spark-dot dot-oversold" title="RSI {v:.0f}">●</span>')
            elif v < 40:
                dots.append(f'<span class="spark-dot dot-cool" title="RSI {v:.0f}">●</span>')
            elif v <= 60:
                dots.append(f'<span class="spark-dot dot-neutral" title="RSI {v:.0f}">●</span>')
            elif v <= 70:
                dots.append(f'<span class="spark-dot dot-warm" title="RSI {v:.0f}">●</span>')
            else:
                dots.append(f'<span class="spark-dot dot-overbought" title="RSI {v:.0f}">●</span>')
    # Add direction arrows between dots based on consecutive RSI values
    spark_html = dots[0]
    for idx in range(1, 6):
        key_prev = f"rsi{idx-1}"
        key_curr = f"rsi{idx}"
        vp = r.get(key_prev)
        vc = r.get(key_curr)
        if vp is not None and vc is not None and not (isinstance(vp, float) and math.isnan(vp)) and not (isinstance(vc, float) and math.isnan(vc)):
            if float(vc) > float(vp):
                arrow = '<span class="spark-arrow arrow-up">↗</span>'
            elif float(vc) < float(vp):
                arrow = '<span class="spark-arrow arrow-down">↘</span>'
            else:
                arrow = '<span class="spark-arrow arrow-flat">→</span>'
        else:
            arrow = '<span class="spark-arrow arrow-flat">→</span>'
        spark_html += arrow + dots[idx]
    rsi0_val = r.get("rsi0")
    rsi0_str = f"{float(rsi0_val):.0f}" if rsi0_val is not None and not (isinstance(rsi0_val, float) and math.isnan(rsi0_val)) else "—"
    return f'<span class="sparkline">{spark_html}<span class="spark-label">{rsi0_str}</span></span>'

def render_rsi_cell(r):
    """RSI cell: color-coded (green/amber/red) plus momentum arrow."""
    rsi = r.get("rsi0")
    if rsi is None or (isinstance(rsi, float) and math.isnan(rsi)):
        return '<td class="rsi-td">—</td>'
    v = float(rsi)
    if 30 <= v < 40:
        cls = "rsi-opportunity"
        label = "🔥 OPPORTUNITY"
    elif 40 <= v < 60:
        cls = "rsi-neutral"
        label = ""
    elif v >= 60:
        cls = "rsi-overbought"
        label = "⚠️ OVERBOUGHT"
    else:
        cls = "rsi-opportunity"
        label = ""
    delta = r.get("rsi_delta_1")
    arrow = ""
    if delta is not None and not (isinstance(delta, float) and math.isnan(delta)):
        d = float(delta)
        if d > 0:
            arrow = ' <span class="mom-arrow mom-up" title="RSI rising">▲</span>'
        elif d < 0:
            arrow = ' <span class="mom-arrow mom-down" title="RSI falling">▼</span>'
    return f'<td class="rsi-td {cls}"><span class="rsi-val">{v:.0f}</span>{arrow}<small>{label}</small></td>'

def render_rank_badge(r):
    """🏆 badge for #1 in sector."""
    rank = r.get("rank_in_sector")
    if rank == 1:
        return '<span class="rank-badge" title="#1 in sector">🏆</span>'
    return ""

def finviz_url(ticker: str) -> str:
    return "https://finviz.com/stock?t=" + urllib.parse.quote(str(ticker).strip().upper(), safe=".-")

def ticker_link(ticker: str, cls: str = "ticker-link") -> str:
    safe_ticker = html.escape(str(ticker).strip().upper())
    safe_href = html.escape(finviz_url(ticker), quote=True)
    return f'<a class="{cls}" href="{safe_href}" target="_blank" rel="noopener noreferrer">{safe_ticker}</a>'
