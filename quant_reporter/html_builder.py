import pandas as pd
import numpy as np
import plotly.io as pio
from io import StringIO


def _build_swot_html(swot_data):
    """Generate SWOT HTML cards from portfolio metrics dict. Returns raw HTML string."""
    tickers = swot_data.get("tickers", [])
    benchmark = swot_data.get("benchmark", "SPY")
    portfolio_dict = swot_data.get("portfolio_dict") or swot_data.get("allocation_weights") or {}
    sector_map = swot_data.get("sector_map", {})
    pr_metrics = swot_data.get("pr_metrics", {})
    asset_corr_html_str = swot_data.get("asset_corr_html", "")

    _is_indian = any(t.upper().endswith((".NS", ".BO")) for t in tickers)
    _market_listing = "NSE-listed equities" if _is_indian else "US-listed equities"
    _market_name = "Indian markets" if _is_indian else "US markets"

    def _pct(key):
        v = swot_data.get(key)
        if v is None:
            return None
        try:
            return float(str(v).replace("%", "").replace(",", "").strip()) / 100
        except Exception:
            return None

    def _flt(key):
        v = swot_data.get(key)
        if v is None:
            return None
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return None

    cagr_asset = _pct("cagr_asset")
    cagr_bench = _pct("cagr_bench")
    asset_vol  = _pct("volatility_asset")
    bench_vol  = _pct("volatility_bench")
    sharpe     = _flt("sharpe")
    sortino    = _flt("sortino")
    calmar     = _flt("calmar")
    max_dd     = _pct("max_drawdown")
    beta       = _flt("beta")
    alpha      = _pct("alpha")
    var_95     = _pct("var_95")

    # ── Parse correlation table ─────────────────────────────────────────
    corr_pairs = {}
    if asset_corr_html_str:
        try:
            cdf = pd.read_html(StringIO(asset_corr_html_str))[0]
            for _, row in cdf.iterrows():
                try:
                    corr_pairs[str(row.iloc[0])] = float(row.iloc[-1])
                except Exception:
                    pass
        except Exception:
            pass
    high_corr = {t: v for t, v in corr_pairs.items() if v > 0.8}

    # ── Composition analysis ────────────────────────────────────────────
    sectors = list(set(sector_map.values())) if sector_map else []
    sector_wts = {}
    for t, w in portfolio_dict.items():
        s = sector_map.get(t, "Other")
        sector_wts[s] = sector_wts.get(s, 0) + w
    dom_sector = max(sector_wts.items(), key=lambda x: x[1]) if sector_wts else ("N/A", 0)
    top3 = sorted(portfolio_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    max_wt = max(portfolio_dict.values()) if portfolio_dict else 0
    tech_set = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMD", "QQQ", "AVGO", "TSM"}
    has_ai = bool(set(t.upper() for t in tickers) & {"NVDA", "QQQ"})
    tech_alloc = sum(w for t, w in portfolio_dict.items() if t.upper() in tech_set)

    # ════════════════════════════════════════════════════════════════════
    # Build bullet points
    # ════════════════════════════════════════════════════════════════════

    # STRENGTHS
    S = []
    if cagr_asset is not None and cagr_bench is not None and cagr_asset > cagr_bench:
        S.append(f"Portfolio CAGR of {cagr_asset:.2%} outperforms {benchmark} benchmark of {cagr_bench:.2%}.")
    if sharpe is not None and sharpe > 0.5:
        S.append(f"Positive risk-adjusted return with Sharpe ratio of {sharpe:.2f}.")
    elif sharpe is not None and sharpe > 0:
        S.append(f"Sharpe ratio is positive ({sharpe:.2f}), indicating returns exceed the risk-free rate per unit of risk.")
    if sortino is not None and sortino > 0.7:
        S.append(f"Strong downside protection with Sortino ratio of {sortino:.2f}.")
    if len(tickers) >= 3:
        S.append(f"Diversification across {len(tickers)} assets ({', '.join(tickers)}).")
    if calmar is not None and calmar > 0.5:
        S.append(f"Calmar ratio of {calmar:.2f} shows returns compensate for drawdown risk.")
    if len(S) < 3 and top3:
        S.append(f"Top holdings: {', '.join(f'{t} ({w:.1%})' for t,w in top3)}.")
    _s_fallbacks = [
        f"Portfolio consists of {len(tickers)} liquid, {_market_listing} with deep historical data.",
        f"All holdings are exchange-traded with transparent daily pricing in {_market_name}.",
        f"Portfolio benefits from broad institutional coverage across {len(tickers)} names.",
    ]
    for _fb in _s_fallbacks:
        if len(S) >= 3:
            break
        S.append(_fb)

    # WEAKNESSES
    W = []
    if beta is not None and beta > 1.2:
        W.append(f"High market sensitivity &mdash; beta of {beta:.2f} indicates {(beta-1)*100:.0f}% more volatility than the benchmark.")
    if max_dd is not None and max_dd < -0.20:
        W.append(f"Significant drawdown risk of {max_dd:.2%} &mdash; portfolio experienced substantial peak-to-trough decline.")
    if alpha is not None and alpha < 0:
        W.append(f"Negative risk-adjusted alpha of {alpha:.2%} &mdash; underperforms on a CAPM basis after adjusting for beta.")
    if high_corr:
        for t, v in sorted(high_corr.items(), key=lambda x: x[1], reverse=True)[:2]:
            W.append(f"High correlation between {t} and benchmark ({v:.2f}) reduces diversification benefit.")
    if asset_vol is not None and bench_vol is not None and asset_vol > bench_vol + 0.02:
        W.append(f"Portfolio volatility ({asset_vol:.2%}) exceeds benchmark volatility ({bench_vol:.2%}) by {(asset_vol-bench_vol):.2%}.")
    if len(W) < 3 and max_wt > 0.35:
        W.append(f"Concentration risk: largest position at {max_wt:.1%} weight.")
    if len(W) < 3 and var_95 is not None:
        W.append(f"Daily Value-at-Risk (95%) is {var_95:.2%} &mdash; a 1-in-20 day loss could reach this level.")
    if len(W) < 3 and len(tickers) < 5:
        W.append(f"Narrow universe of only {len(tickers)} assets limits diversification benefits.")
    _w_fallbacks = [
        "No fixed-income or alternative assets in the portfolio to buffer equity drawdowns.",
        "Portfolio lacks commodity or gold exposure as an inflation hedge.",
        "No explicit currency hedging for international holdings.",
    ]
    for _fb in _w_fallbacks:
        if len(W) >= 3:
            break
        W.append(_fb)

    # OPPORTUNITIES
    O = []
    if has_ai:
        ai_names = [t for t in tickers if t.upper() in {"NVDA", "QQQ"}]
        O.append(f"AI and semiconductor sector exposure through {' and '.join(ai_names)} positions &mdash; growth potential in AI infrastructure.")
    if tech_alloc > 0.50:
        O.append(f"Technology sector concentration ({tech_alloc:.0%}) offers upside in continued digital transformation.")
    if var_95 is not None and var_95 < -0.02:
        O.append("Tail risk management opportunity &mdash; consider protective puts or collar strategies for hedging.")
    O.append("Walk-forward rebalancing on a quarterly cadence could capture regime changes and improve out-of-sample alpha.")
    if len(sectors) < 4:
        O.append("Adding uncorrelated sectors (Healthcare, Utilities, REITs) would improve the efficient frontier.")
    if sharpe is not None and sharpe > 0 and beta is not None and beta > 1:
        O.append("A lower-beta variant could achieve similar risk-adjusted returns with reduced drawdown exposure.")
    _o_fallbacks = [
        "Options overlays (e.g., covered calls) could enhance yield and reduce effective portfolio volatility.",
        "Factor tilts (value, momentum, quality) could improve risk-adjusted returns.",
        "Systematic rebalancing triggers based on drift thresholds would enforce discipline.",
    ]
    for _fb in _o_fallbacks:
        if len(O) >= 3:
            break
        O.append(_fb)

    # THREATS
    T = []
    if beta is not None and beta > 1.2:
        T.append(f"Amplified drawdown in market downturns due to high beta ({beta:.2f}x) &mdash; a 20% correction implies ~{beta*20:.0f}% portfolio loss.")
    if asset_vol is not None and bench_vol is not None and asset_vol > bench_vol + 0.05:
        T.append(f"Portfolio volatility ({asset_vol:.2%}) significantly exceeds benchmark ({bench_vol:.2%}) &mdash; elevated risk in choppy markets.")
    if max_wt > 0.35:
        T.append(f"Concentration risk &mdash; single asset at {max_wt:.1%} creates idiosyncratic event vulnerability.")
    if dom_sector[1] > 0.60:
        T.append(f"Sector concentration ({dom_sector[0]} at {dom_sector[1]:.0%}) creates vulnerability to regulation or macro headwinds.")
    if high_corr:
        T.append(f"High benchmark correlation ({', '.join(f'{t} &rho;={v:.2f}' for t,v in list(high_corr.items())[:3])}) offers limited diversification in a crash.")
    T.append("Historical optimization may not predict future regimes &mdash; structural market shifts could invalidate assumptions.")
    _t_fallbacks = [
        "Rising interest rates could compress equity valuations, especially for growth-oriented holdings.",
        "Geopolitical shocks or trade policy changes could disrupt sector performance.",
        "Liquidity contraction in risk-off environments may widen bid-ask spreads on smaller positions.",
    ]
    for _fb in _t_fallbacks:
        if len(T) >= 3:
            break
        T.append(_fb)

    # ════════════════════════════════════════════════════════════════════
    # Render 2×2 HTML grid
    # ════════════════════════════════════════════════════════════════════
    cards = [
        ("Strengths",     S, "#4ade80", "#0a1f0f"),
        ("Weaknesses",    W, "#fbbf24", "#1f1505"),
        ("Opportunities", O, "#60a5fa", "#040f1f"),
        ("Threats",       T, "#f87171", "#1f0507"),
    ]

    html = '<div style="display:flex;flex-wrap:wrap;gap:20px;margin-top:16px;">'
    for heading, bullets, accent, bg in cards:
        li_items = "".join(
            f'<li style="margin-bottom:8px;padding:8px 12px;border-radius:6px;'
            f'background:rgba(255,255,255,0.04);border-left:3px solid {accent};'
            f'list-style:none;color:#e2e8f0;font-size:0.9rem;line-height:1.5;">{b}</li>'
            for b in bullets
        )
        html += (
            f'<div style="flex:1 1 45%;min-width:300px;background:{bg};'
            f'border:1px solid {accent}30;border-radius:12px;padding:20px 22px;'
            f'box-shadow:0 4px 20px rgba(0,0,0,0.3);">'
            f'<div style="font-size:1.1rem;font-weight:800;color:{accent};'
            f'margin-bottom:12px;letter-spacing:0.5px;">{heading}</div>'
            f'<ul style="margin:0;padding:0;">{li_items}</ul>'
            f'</div>'
        )
    html += '</div>'
    
    per_asset_swot = swot_data.get("per_asset_swot", {})
    if per_asset_swot:
        html += '<h3 style="margin-top:24px; color:var(--text-color);">Individual Asset Analysis</h3>'
        html += '<div style="display:flex;flex-wrap:wrap;gap:16px;">'
        for ticker, data in per_asset_swot.items():
            def _sf(val, default=None):
                try:
                    return float(val) if val not in (None, '', 'None') else default
                except (ValueError, TypeError):
                    return default

            s_b, w_b, o_b, t_b = [], [], [], []
            ret = _sf(data.get('cagr_1y'))
            if ret is not None: s_b.append(f"1Y Return: {ret:.2%}")
            beta_a = _sf(data.get('beta'))
            if beta_a is not None: s_b.append(f"Beta vs Bench: {beta_a:.2f}")
            corr_p = _sf(data.get('corr_port'))
            if corr_p is not None: s_b.append(f"Port Corr: {corr_p:.2f}")
            vol = _sf(data.get('volatility'))
            if vol is not None: w_b.append(f"Volatility: {vol:.2%}")
            mdd = _sf(data.get('max_drawdown'))
            if mdd is not None: w_b.append(f"Max DD: {mdd:.2%}")
            corr_b = _sf(data.get('corr_bench'))
            if corr_b is not None and corr_b > 0.9: w_b.append(f"High Bench Corr: {corr_b:.2f}")
            sec = data.get('sector')
            if sec and sec != "Other": o_b.append(f"Sector: {sec}")
            if ret is not None and ret > 0.2: o_b.append("Strong Momentum (>20%)")
            wt = _sf(data.get('weight', 0), default=0)
            if wt > 0.3: t_b.append(f"Concentration: {wt:.1%}")
            if beta_a is not None and beta_a > 1.3: t_b.append(f"High Beta Risk: {beta_a:.2f}")
            
            if not s_b: s_b.append("Core Holding")
            if not w_b: w_b.append("Standard Risk Profile")
            if not o_b: o_b.append("Long-term Growth")
            if not t_b: t_b.append("Market Correlation")

            def _render_mini_list(color, items):
                return "".join(f"<div style='font-size:0.85rem; margin-bottom:4px; border-left: 2px solid {color}; padding-left: 6px; color:var(--text-color-muted);'>{item}</div>" for item in items[:2])

            html += f"""
            <div style="flex: 1 1 200px; border-radius:12px; padding:16px; background:var(--card-color); border:1px solid var(--border-color); box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="font-weight:bold; font-size:1.1rem; color:var(--text-color);">{ticker}</span>
                    <span style="font-size:0.8rem; padding:2px 8px; border-radius:12px; background:#e2e8f0; color:#475569;">{wt:.1%} Wt</span>
                </div>
                <div style="margin-bottom:8px;">
                    <div style="font-size:0.8rem; font-weight:bold; color:#4ade80; margin-bottom:4px;">Strengths</div>
                    {_render_mini_list('#4ade80', s_b)}
                </div>
                <div style="margin-bottom:8px;">
                    <div style="font-size:0.8rem; font-weight:bold; color:#fbbf24; margin-bottom:4px;">Weaknesses</div>
                    {_render_mini_list('#fbbf24', w_b)}
                </div>
                <div style="margin-bottom:8px;">
                    <div style="font-size:0.8rem; font-weight:bold; color:#60a5fa; margin-bottom:4px;">Opportunities</div>
                    {_render_mini_list('#60a5fa', o_b)}
                </div>
                <div>
                    <div style="font-size:0.8rem; font-weight:bold; color:#f87171; margin-bottom:4px;">Threats</div>
                    {_render_mini_list('#f87171', t_b)}
                </div>
            </div>"""
            
        html += '</div>'
        
    return html


def generate_html_report(sections, title="Quantitative Report", filename="report.html"):
    """
    Generates a flexible, cross-browser HTML report from a list of sections.
    """
    print("Assembling HTML report...")

    sidebar_html = ""
    main_content_html = ""
    toc_links = []
    js_added = False  

    for i, section in enumerate(sections):
        section_id = f"section-{i}"
        toc_links.append(f'<li><a href="#{section_id}">{section["title"]}</a></li>')

        # --- Section Container ---
        main_content_html += f'<div id="{section_id}" class="report-section">'
        main_content_html += f'<h1>{section["title"]}</h1>'
        if "description" in section:
            main_content_html += f'<p>{section["description"]}</p>'

        # --- Inject Sidebar Metrics as Top Grid ---
        if "sidebar" in section and section["sidebar"]:
            main_content_html += '<div class="metrics-grid">'
            for item in section["sidebar"]:
                main_content_html += '<div class="metrics-card">'
                main_content_html += f'<h2>{item["title"]}</h2>'
                if item["type"] == "metrics":
                    df = pd.DataFrame.from_dict(item["data"], orient='index', columns=['Value'])
                    main_content_html += df.to_html(header=False, classes='metrics-table')
                elif item["type"] == "table_html":
                    main_content_html += item["data"]
                main_content_html += '</div>'
            main_content_html += '</div>'

        for item in section["main_content"]:
            item_class = "plot-item" if item["type"] == "plot" else "table-item"
            main_content_html += f'<div class="{item_class}">'
            if "title" in item:
                main_content_html += f'<h2>{item["title"]}</h2>'

            if item["type"] == "plot":
                if not js_added:
                    js_load_strategy = 'cdn'
                    js_added = True
                else:
                    js_load_strategy = False

                main_content_html += pio.to_html(
                    item["data"],
                    full_html=False,
                    include_plotlyjs=js_load_strategy
                )

            elif item["type"] == "table_html":
                main_content_html += item["data"]

            elif item["type"] == "metrics_grid":
                main_content_html += '<div class="metrics-grid">'
                for name, data in item["data"].items():
                    main_content_html += '<div class="metrics-card">'
                    main_content_html += f"<h2>{name} Portfolio</h2>"

                    weights_df = pd.DataFrame.from_dict(
                        data['weights_dict'], orient='index', columns=['Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'].map(lambda x: f"{x:.2%}")
                    main_content_html += "<h3>Optimal Weights</h3>"
                    main_content_html += weights_df.to_html(classes='metrics-table')

                    metrics_df = pd.DataFrame.from_dict(data['metrics'], orient='index', columns=['Value'])
                    main_content_html += "<h3>Performance Metrics</h3>"
                    main_content_html += metrics_df.to_html(header=False, classes='metrics-table')
                    main_content_html += "</div>"
                main_content_html += '</div>'

            elif item["type"] == "swot_placeholder":
                main_content_html += _build_swot_html(item.get("data", {}))
        main_content_html += '</div>'

    # --- Build Final HTML ---
    html_template = f"""
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>{title}</title>
        <style>
            /* Normalize + base styles */
            *, *::before, *::after {{ box-sizing: border-box; }}
            html, body {{ margin: 0; padding: 0; }}
            
            :root {{
                --bg-color: #0b0e14;
                --card-color: #151b23;
                --text-color: #e2e8f0;
                --text-color-muted: #94a3b8;
                --border-color: #30363d;
                --accent-color: #00d4ff;
                --shadow-subtle: 0 4px 12px rgba(0, 0, 0, 0.3);
                --shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.5);
                --border-radius: 12px;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: var(--bg-color); color: var(--text-color); margin: 0; padding: 16px;
            }}
            .container {{
                display: flex; flex-direction: column; align-items: stretch;
                justify-content: flex-start; max-width: 1400px; margin: 0 auto; gap: 30px; width: 100%;
            }}
            h1 {{
                text-align: center; color: var(--text-color); font-size: 2.5em;
                font-weight: 700; margin-bottom: 24px; width: 100%;
                background: -webkit-linear-gradient(45deg, #00d4ff, #bf7fff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .report-section {{
                background-color: var(--bg-color);
                margin-bottom: 40px; padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
                width: 100%;
                display: flex; flex-direction: column; gap: 24px;
            }}
            .report-section h1 {{
                text-align: left; font-size: 2em; margin-bottom: 10px; background: none; -webkit-text-fill-color: var(--text-color);
            }}
            h2 {{ color: var(--accent-color); padding-bottom: 5px; font-weight: 600; font-size: 1.4rem; }}
            h3 {{ color: var(--text-color); font-weight: 500; }}
            
            .toc {{
                background-color: var(--card-color); border: 1px solid var(--border-color);
                border-radius: var(--border-radius); padding: 20px; margin-bottom: 10px;
                box-shadow: var(--shadow-subtle); width: 100%;
                overflow-x: auto;
            }}
            .toc ul {{ list-style-type: none; padding-left: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 10px; }}
            .toc li {{ margin-bottom: 0; }}
            .toc a {{ text-decoration: none; color: var(--text-color-muted); font-weight: 500; transition: color 0.2s ease; display: inline-block; padding: 6px 12px; border-radius: 6px; background-color: rgba(255,255,255,0.02); border: 1px solid var(--border-color); }}
            .toc a:hover {{ color: var(--accent-color); background-color: rgba(0, 212, 255, 0.05); border-color: var(--accent-color); }}
            
            .metrics-table {{
                width: 100%; border-collapse: separate; border-spacing: 0; margin-bottom: 20px;
                border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color);
                table-layout: fixed;
            }}
            .metrics-table th {{ background-color: #1e2631; padding: 12px 14px; text-align: left; color: var(--text-color); font-weight: 600; }}
            .metrics-table tr:nth-child(even) {{ background-color: rgba(255,255,255,0.02); }}
            .metrics-table td {{ padding: 12px 14px; border-bottom: 1px solid var(--border-color); word-wrap: break-word; }}
            .metrics-table tr:last-child td {{ border-bottom: none; }}
            .metrics-table td:first-child {{ font-weight: 500; color: var(--text-color-muted); width: 60%; }}
            .metrics-table td:not(:first-child) {{ text-align: right; font-weight: 600; color: var(--text-color); }}
            
            .plot-item, .table-item {{
                background-color: var(--card-color); border: 1px solid var(--border-color);
                border-radius: var(--border-radius); padding: 24px;
                box-shadow: var(--shadow-subtle);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                overflow-x: auto; width: 100%;
                margin-bottom: 24px;
            }}
            .plot-item:hover, .table-item:hover, .metrics-card:hover {{
                box-shadow: var(--shadow-hover);
            }}
            .metrics-grid {{
                display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; margin-top: 10px; margin-bottom: 24px; width: 100%;
            }}
            .metrics-card {{
                background-color: var(--card-color); border: 1px solid var(--border-color);
                border-radius: 8px; padding: 20px; box-shadow: var(--shadow-subtle);
            }}
            .report-section h1, .report-section h2, .report-section h3 {{
                word-wrap: break-word;
                margin-top: 0;
            }}
            iframe, div.plotly-graph-div {{
                width: 100% !important;
                min-height: 450px;
                display: block;
            }}
        </style>
    </head>
    <body>
        <noscript>
            <div style="background: #ffdddd; padding: 10px; border: 1px solid red;">
                ⚠️ This report requires JavaScript to display interactive plots. Please enable it.
            </div>
        </noscript>
        <div class="container">
            <div class="toc">
                <h2>Quick Navigation</h2>
                <ul>{ "".join(toc_links) }</ul>
            </div>
            {main_content_html}
        </div>
    </body>
    </html>
    """

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"✅ Report successfully generated: {filename}")
    except Exception as e:
        print(f"❌ Error writing HTML file: {e}")