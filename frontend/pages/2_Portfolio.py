import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import threading
import os
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Import quant_reporter
try:
    import quant_reporter as qr
    import yfinance as yf

    # --- MONKEY PATCH FOR quant_reporter TO HANDLE yfinance>=0.2.40 SERIES OBJECTS ---
    if hasattr(qr, 'opt_core'):
        def patched_get_risk_free_rate():
            try:
                tbill = yf.download("^IRX", period="5d")
                if tbill is None or tbill.empty:
                    raise Exception("^IRX download failed")
                latest_rate = float(np.ravel(tbill['Close'].iloc[-1])[0]) / 100
                if not 0 <= latest_rate <= 0.2:
                    raise Exception("Rate unrealistic.")
                return latest_rate
            except Exception:
                return 0.06
        qr.opt_core.get_risk_free_rate = patched_get_risk_free_rate

        def patched_calculate_rolling_returns(cumulative_df):
            periods = {'1-Year': 252, '3-Year': 252*3, '5-Year': 252*5}
            rolling_returns = {}
            for name, days in periods.items():
                if len(cumulative_df) > days:
                    years = days / 252
                    try:
                        end_val = float(np.ravel(cumulative_df.iloc[-1])[0])
                        start_val = float(np.ravel(cumulative_df.iloc[-days-1])[0])
                        val = (end_val / start_val) ** (1 / years) - 1
                        rolling_returns[name] = {0: f"{val:.2%}"}
                    except Exception:
                        rolling_returns[name] = {0: "N/A"}
                else:
                    rolling_returns[name] = {0: "N/A"}
            return pd.DataFrame.from_dict(rolling_returns, orient='index')

        qr.opt_core.calculate_rolling_returns = patched_calculate_rolling_returns

        # CRITICAL: also patch the reference inside combined_report module
        # because it imports the function directly into its own namespace
        try:
            import quant_reporter.combined_report as _qr_combined
            _qr_combined.calculate_rolling_returns = patched_calculate_rolling_returns
        except Exception:
            pass
    # ---------------------------------------------------------------------------------

    QUANT_REPORTER_AVAILABLE = True
except ImportError:
    QUANT_REPORTER_AVAILABLE = False

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Portfolio - QuantVision", layout="wide")

st.title("Portfolio Manager")

def _render_allocation_charts(allocation):
    if not allocation:
        st.warning("No allocation data.")
        return
    col_chart, col_table = st.columns(2)
    with col_chart:
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation.keys()),
            values=list(allocation.values()),
            hole=0.4
        )])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, width='stretch')
    with col_table:
        df = pd.DataFrame(list(allocation.items()), columns=["Ticker", "Weight"])
        df["Weight"] = df["Weight"].apply(lambda x: f"{x:.1%}")
        st.table(df.set_index("Ticker"))

def _render_allocation(result, tickers):
    method = result.get("method", "standard")

    if method == "bl":
        # --- LSTM Views Table with animated spinners ---
        st.subheader("LSTM Model Views")
        st.caption(
            "Confidence is derived from each model's directional accuracy. "
            "Evolved = GA-optimized. Optimizing = GA running in background."
        )

        view_details = result.get("view_details", [])

        # Check for any queued/optimizing tickers
        has_optimizing = any(
            v.get("model_source") in ("queued", "default")
            for v in view_details
        )

        # Build custom HTML table with animated spinner for optimizing rows
        table_html = """
        <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
            margin: 0;
            padding: 8px;
            font-family: sans-serif;
        }
        .views-table {
            width: 100%;
            min-width: 1000px;
            border-collapse: collapse;
            font-size: 14px;
            margin-bottom: 16px;
        }
        .views-table th {
            text-align: left;
            padding: 8px 12px;
            border-bottom: 2px solid #333;
            color: #aaa;
            font-weight: 500;
        }
        .views-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #222;
            color: #fafafa;
        }
        .evolved-badge {
            color: #4CAF50;
            font-weight: 600;
        }
        .optimizing-badge {
            color: #FFA500;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .spinner {
            width: 12px;
            height: 12px;
            border: 2px solid #FFA500;
            border-top-color: transparent;
            border-radius: 50%;
            display: inline-block;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        </style>
        <table class="views-table">
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Current Price</th>
                <th>Predicted Price</th>
                <th>Expected Return</th>
                <th>Confidence</th>
                <th>Model</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
        """

        for v in view_details:
            ret = v.get("expected_return", 0)
            ret_class = "positive" if ret >= 0 else "negative"
            ret_str = f"{ret:+.2f}%"

            source = v.get("model_source", "default")
            if source == "evolved":
                model_cell = f'<span class="evolved-badge">Evolved</span>'
            else:
                model_cell = (
                    '<span class="optimizing-badge">'
                    '<span class="spinner"></span>Optimizing...'
                    '</span>'
                )

            accuracy = v.get("accuracy")
            accuracy_str = f"{accuracy}%" if accuracy else "—"

            table_html += f"""
            <tr>
                <td><strong>{v['symbol']}</strong></td>
                <td>${v['current_price']}</td>
                <td>${v['predicted_price']}</td>
                <td class="{ret_class}">{ret_str}</td>
                <td>{v['confidence']}%</td>
                <td>{model_cell}</td>
                <td>{accuracy_str}</td>
            </tr>
            """

        table_html += "</tbody></table>"
        components.html(
            table_html,
            height=max(len(view_details) * 55 + 150, 300),
            scrolling=True
        )

        # Store optimizing status for auto-refresh
        st.session_state["has_optimizing"] = has_optimizing

        # Caption for queued tickers
        queued = [
            v["symbol"] for v in view_details
            if v.get("model_source") in ("queued", "default")
        ]
        if queued:
            st.caption(
                f"Optimizing in background: {', '.join(queued)}. "
                f"Page will refresh automatically when ready."
            )

        st.divider()
        st.subheader("Recommended Allocation")
        _render_allocation_charts(result.get("allocation", {}))
        st.caption("Method: AI-Driven Black-Litterman")

    else:
        # Standard fallback
        st.info("Using risk-based optimization (BL unavailable for these tickers).")
        st.subheader("Recommended Allocation")
        _render_allocation_charts(result.get("allocation", {}))
        st.caption("Method: Risk-Based (HRP/CVaR/Sharpe)")

    # AI Reasoning
    reasoning = result.get("reasoning", "")
    if reasoning:
        st.info(f"**AI Analysis:**\n\n{reasoning}")

def _run_allocation(tickers, risk_tolerance):
    """
    Tries BL first. Falls back to standard optimizer on failure.
    Stores result in session state.
    """
    with st.spinner("Running LSTM predictions and portfolio optimization..."):
        # Try Black-Litterman first
        try:
            bl_resp = requests.post(
                f"{API_URL}/bl-optimize/",
                json={"tickers": tickers},
                timeout=300
            )
            if bl_resp.status_code == 200:
                result = bl_resp.json()
                result["method"] = "bl"
                st.session_state["allocation_result"] = result
                _render_allocation(result, tickers)
                return
        except Exception:
            pass

        # Fallback to standard optimizer
        try:
            std_resp = requests.post(
                f"{API_URL}/optimize/",
                json={"tickers": tickers, "risk_tolerance": risk_tolerance}
            )
            if std_resp.status_code == 200:
                result = std_resp.json()
                result["method"] = "standard"
                st.session_state["allocation_result"] = result
                _render_allocation(result, tickers)
        except Exception as e:
            st.error(f"Optimization failed: {e}")

# Tabs
tab_allocator, tab_bl, tab_report = st.tabs([
    "Portfolio Allocator",
    "Black-Litterman",
    "Deep Report (Backtest)"
])

# --- TAB 1: ALLOCATOR ---
with tab_allocator:
    # Silent auto-queue for default tickers on page load
    _default_tickers = ["AAPL", "TSLA", "SPY", "MSFT", "GOOG", "NVDA", "AMZN"]
    if "auto_queued" not in st.session_state:
        try:
            unoptimized = []
            for _t in _default_tickers:
                _r = requests.get(
                    f"{API_URL}/hpo-status/?symbol={_t}",
                    timeout=2
                )
                if _r.status_code == 200 and not _r.json().get("optimized"):
                    unoptimized.append(_t)
            if unoptimized:
                st.caption(
                    f"Models will be optimized automatically when you generate "
                    f"an allocation for: {', '.join(unoptimized)}"
                )
        except Exception:
            pass
        st.session_state["auto_queued"] = True

    st.markdown("### Portfolio Allocator")
    st.markdown(
        "Generates an optimal allocation using **AI-Driven Black-Litterman** "
        "optimization — combining LSTM price predictions with market equilibrium. "
        "Falls back to risk-based optimization if prediction data is unavailable."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_area(
            "Enter Stock Tickers (comma separated, min 2)",
            "AAPL, GOOG, MSFT, TSLA",
            height=80
        )
    with col2:
        risk_tolerance = st.slider("Risk Tolerance (0=Safe, 1=Risky)", 0.0, 1.0, 0.5)
        st.caption("Used as fallback if BL optimization fails.")

    if st.button("Generate Allocation", type="primary"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            st.session_state["last_tickers"] = tickers
            st.session_state["allocation_result"] = None
            st.session_state["refresh_count"] = 0
            st.session_state["_just_ran_allocation"] = True
            _run_allocation(tickers, risk_tolerance)

    # Re-render allocation result on tab switch (persists across reruns)
    if st.session_state.get("allocation_result") and \
       not st.session_state.get("_just_ran_allocation"):
        _render_allocation(
            st.session_state["allocation_result"],
            st.session_state.get("last_tickers", [])
        )

    # Reset flag after render
    st.session_state["_just_ran_allocation"] = False

    # Auto-refresh when tickers are still being optimized (non-blocking)
    if st.session_state.get("has_optimizing"):
        last_refresh = st.session_state.get("last_refresh_time", 0)
        current_time = time.time()
        if current_time - last_refresh >= 30:
            st.session_state["last_refresh_time"] = current_time
            last_tickers = st.session_state.get("last_tickers", [])
            still_optimizing = False
            for t in last_tickers:
                try:
                    r = requests.get(
                        f"{API_URL}/hpo-status/?symbol={t}",
                        timeout=3
                    )
                    if r.status_code == 200 and not r.json().get("optimized"):
                        still_optimizing = True
                        break
                except Exception:
                    pass
            if not still_optimizing:
                st.session_state["has_optimizing"] = False
            st.rerun()

# --- TAB 2: BLACK-LITTERMAN ---
with tab_bl:
    st.markdown("### Black-Litterman Model")
    st.markdown(
        "Specify your **market views** and confidence levels to compute "
        "Black-Litterman posterior returns and optimal portfolio weights. "
        "BL math is implemented from scratch with NumPy."
    )

    # --- Ticker input ---
    bl_tickers_input = st.text_input(
        "Tickers (comma separated)",
        "AAPL, MSFT, GOOG, AMZN, TSLA, NVDA",
        key="bl_tickers"
    )
    bl_tickers = [t.strip().upper() for t in bl_tickers_input.split(",") if t.strip()]

    if len(bl_tickers) < 2:
        st.warning("Enter at least 2 tickers.")
    else:
        # --- Market weights ---
        st.markdown("#### Market Cap Weights")
        st.caption(
            "Auto-populated from yfinance market caps. "
            "Toggle manual override to edit."
        )

        if "bl_mcap_fetched" not in st.session_state or \
           st.session_state.get("bl_mcap_tickers") != bl_tickers:
            # Fetch market caps
            try:
                import yfinance as yf_bl
                caps = {}
                for t in bl_tickers:
                    try:
                        info = yf_bl.Ticker(t).info
                        cap = info.get("marketCap") or info.get("market_cap")
                        if cap and cap > 0:
                            caps[t] = float(cap)
                    except Exception:
                        pass
                if not caps:
                    caps = {t: 1.0 for t in bl_tickers}
                avg = np.mean(list(caps.values()))
                for t in bl_tickers:
                    if t not in caps:
                        caps[t] = avg
                total = sum(caps.values())
                st.session_state["bl_mcap_weights"] = {t: round(caps[t] / total, 4) for t in bl_tickers}
            except Exception:
                st.session_state["bl_mcap_weights"] = {t: round(1.0 / len(bl_tickers), 4) for t in bl_tickers}
            st.session_state["bl_mcap_fetched"] = True
            st.session_state["bl_mcap_tickers"] = bl_tickers

        manual_override = st.checkbox("Manual override weights", key="bl_manual_weights")

        mcap_weights = {}
        if manual_override:
            cols_w = st.columns(min(len(bl_tickers), 4))
            for i, t in enumerate(bl_tickers):
                with cols_w[i % min(len(bl_tickers), 4)]:
                    default_w = st.session_state["bl_mcap_weights"].get(t, round(1.0 / len(bl_tickers), 4))
                    mcap_weights[t] = st.number_input(
                        t, min_value=0.0, max_value=1.0,
                        value=default_w, step=0.01,
                        key=f"bl_w_{t}"
                    )
            # Normalize
            total_w = sum(mcap_weights.values())
            if total_w > 0:
                mcap_weights = {t: v / total_w for t, v in mcap_weights.items()}
        else:
            mcap_weights = st.session_state.get("bl_mcap_weights", {t: 1.0 / len(bl_tickers) for t in bl_tickers})
            # Display as table
            w_df = pd.DataFrame(
                [(t, f"{w:.2%}") for t, w in mcap_weights.items()],
                columns=["Ticker", "Market Weight"]
            )
            st.dataframe(w_df.set_index("Ticker"), use_container_width=True)

        # --- View Builder ---
        st.markdown("#### Views")
        st.caption("Add your market views. Each view expresses a return expectation with a confidence level.")

        if "bl_views" not in st.session_state:
            st.session_state["bl_views"] = [
                {"type": "absolute", "assets": [bl_tickers[0]], "return_pct": 10.0, "confidence_pct": 65},
            ]

        def _add_view():
            st.session_state["bl_views"].append(
                {"type": "absolute", "assets": [bl_tickers[0]], "return_pct": 5.0, "confidence_pct": 50}
            )

        def _remove_view(idx):
            if len(st.session_state["bl_views"]) > 1:
                st.session_state["bl_views"].pop(idx)

        for vi, view in enumerate(st.session_state["bl_views"]):
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1.5, 2, 1.5, 2, 0.8])
                with c1:
                    vtype = st.selectbox(
                        "Type", ["absolute", "relative"],
                        index=0 if view["type"] == "absolute" else 1,
                        key=f"bl_vtype_{vi}"
                    )
                    st.session_state["bl_views"][vi]["type"] = vtype
                with c2:
                    if vtype == "absolute":
                        asset = st.selectbox(
                            "Asset", bl_tickers,
                            index=bl_tickers.index(view["assets"][0]) if view["assets"][0] in bl_tickers else 0,
                            key=f"bl_vasset_{vi}"
                        )
                        st.session_state["bl_views"][vi]["assets"] = [asset]
                    else:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            a1 = st.selectbox(
                                "Long", bl_tickers,
                                index=0, key=f"bl_va1_{vi}"
                            )
                        with col_b:
                            a2 = st.selectbox(
                                "Short", bl_tickers,
                                index=min(1, len(bl_tickers) - 1), key=f"bl_va2_{vi}"
                            )
                        st.session_state["bl_views"][vi]["assets"] = [a1, a2]
                with c3:
                    ret = st.number_input(
                        "Return %", value=view["return_pct"],
                        min_value=-100.0, max_value=500.0, step=0.5,
                        key=f"bl_vret_{vi}"
                    )
                    st.session_state["bl_views"][vi]["return_pct"] = ret
                with c4:
                    conf = st.slider(
                        "Confidence %", 1, 99, int(view["confidence_pct"]),
                        key=f"bl_vconf_{vi}"
                    )
                    st.session_state["bl_views"][vi]["confidence_pct"] = conf
                with c5:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.button("X", key=f"bl_vdel_{vi}", on_click=_remove_view, args=(vi,))

        st.button("+ Add View", on_click=_add_view, key="bl_add_view")

        # --- Parameters ---
        st.markdown("#### Parameters")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            bl_tau = st.slider("Tau (uncertainty scalar)", 0.01, 0.20, 0.05, 0.01, key="bl_tau")
        with pcol2:
            bl_rf = st.number_input("Risk-Free Rate", 0.0, 0.15, 0.02, 0.005, key="bl_rf")

        # --- Run ---
        if st.button("Run Black-Litterman Analysis", type="primary", key="bl_run"):
            with st.spinner("Running Black-Litterman analysis..."):
                try:
                    payload = {
                        "tickers": bl_tickers,
                        "market_weights": mcap_weights,
                        "views": st.session_state["bl_views"],
                        "tau": bl_tau,
                        "risk_free_rate": bl_rf,
                    }
                    resp = requests.post(
                        f"{API_URL}/black-litterman/",
                        json=payload,
                        timeout=120
                    )
                    if resp.status_code == 200:
                        st.session_state["bl_result"] = resp.json()
                    else:
                        err = resp.json().get("error", "Unknown error")
                        st.error(f"Analysis failed: {err}")
                        st.session_state["bl_result"] = None
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.session_state["bl_result"] = None

        # --- Results ---
        bl_res = st.session_state.get("bl_result")
        if bl_res and not bl_res.get("error"):
            st.divider()
            st.markdown("### Results")
            st.caption(
                f"Delta (risk aversion): {bl_res['delta']} | "
                f"Tau: {bl_res['tau']} | "
                f"Risk-free rate: {bl_res['risk_free_rate']}"
            )

            # --- Side-by-side bar chart: Eq weights vs BL weights ---
            st.subheader("Equilibrium vs BL Optimal Weights")
            eq_w = bl_res["equilibrium_weights"]
            bl_w = bl_res["bl_weights"]
            tks = bl_res["tickers"]

            fig_weights = go.Figure()
            fig_weights.add_trace(go.Bar(
                name="Equilibrium (Market)",
                x=tks,
                y=[eq_w.get(t, 0) for t in tks],
                marker_color="#636EFA",
            ))
            fig_weights.add_trace(go.Bar(
                name="BL Optimal",
                x=tks,
                y=[bl_w.get(t, 0) for t in tks],
                marker_color="#00CC96",
            ))
            fig_weights.update_layout(
                barmode="group",
                yaxis_title="Weight",
                xaxis_title="Ticker",
                margin=dict(t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_weights, use_container_width=True)

            # --- Returns table ---
            st.subheader("Return Comparison")
            ret_table = bl_res["return_table"]
            ret_df = pd.DataFrame(ret_table)
            ret_df.columns = ["Ticker", "Pi (Eq Return %)", "BL Return %", "Difference %", "Optimal Weight"]
            ret_df["Optimal Weight"] = ret_df["Optimal Weight"].apply(lambda x: f"{x:.2%}")

            def _color_diff(val):
                try:
                    v = float(val)
                    if v > 0:
                        return "color: #4CAF50"
                    elif v < 0:
                        return "color: #f44336"
                except (ValueError, TypeError):
                    pass
                return ""

            st.dataframe(
                ret_df.set_index("Ticker").style.applymap(
                    _color_diff, subset=["Difference %"]
                ),
                use_container_width=True
            )

            # --- View decomposition chart ---
            st.subheader("View Portfolio Decomposition")
            view_decomp = bl_res.get("view_decomposition", [])
            if view_decomp:
                # Stacked bar: market portfolio + each view contribution
                decomp_fig = go.Figure()

                # Market portfolio base
                decomp_fig.add_trace(go.Bar(
                    name="Market Portfolio",
                    x=tks,
                    y=[eq_w.get(t, 0) for t in tks],
                    marker_color="#636EFA",
                ))

                colors = ["#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
                for idx, vd in enumerate(view_decomp):
                    lam = vd["lambda"]
                    vw = vd["weights"]
                    decomp_fig.add_trace(go.Bar(
                        name=f"View {idx + 1} (λ={lam:.4f})",
                        x=tks,
                        y=[vw.get(t, 0) * lam for t in tks],
                        marker_color=colors[idx % len(colors)],
                    ))

                decomp_fig.update_layout(
                    barmode="stack",
                    yaxis_title="Weight Contribution",
                    xaxis_title="Ticker",
                    margin=dict(t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(decomp_fig, use_container_width=True)

            # --- Implied views (reverse BL) ---
            st.subheader("Implied Views (Reverse Black-Litterman)")
            impl_views = bl_res.get("implied_views", [])
            if impl_views:
                iv_rows = []
                for iv in impl_views:
                    assets_str = " vs ".join(iv["assets"]) if iv["type"] == "relative" else iv["assets"][0]
                    iv_rows.append({
                        "View": f"View {iv['view_index'] + 1}",
                        "Type": iv["type"].title(),
                        "Assets": assets_str,
                        "Stated Return %": iv["stated_return_pct"],
                        "Implied Return %": iv["implied_return_pct"],
                    })
                iv_df = pd.DataFrame(iv_rows)
                st.dataframe(iv_df.set_index("View"), use_container_width=True)

            # --- Omega confidence visualization ---
            st.subheader("Omega Diagonal (View Uncertainty)")
            omega_vals = bl_res.get("omega_diagonal", [])
            if omega_vals:
                view_labels = [f"View {i+1}" for i in range(len(omega_vals))]
                fig_omega = go.Figure()
                fig_omega.add_trace(go.Bar(
                    x=view_labels,
                    y=omega_vals,
                    marker_color=["#FFA15A" if v > np.median(omega_vals) else "#00CC96" for v in omega_vals],
                    text=[f"{v:.6f}" for v in omega_vals],
                    textposition="outside",
                ))
                fig_omega.update_layout(
                    yaxis_title="Omega (uncertainty)",
                    xaxis_title="View",
                    margin=dict(t=30, b=30),
                )
                st.plotly_chart(fig_omega, use_container_width=True)
                st.caption("Lower omega = higher confidence in the view. Calibrated via Idzorek's method.")

            # --- Portfolio stats ---
            st.subheader("Portfolio Statistics")
            pstats = bl_res["portfolio_stats"]
            mstats = bl_res["market_stats"]
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric(
                    "Expected Return",
                    f"{pstats['expected_return']:.2f}%",
                    delta=f"{pstats['expected_return'] - mstats['expected_return']:+.2f}% vs market"
                )
            with sc2:
                st.metric(
                    "Volatility",
                    f"{pstats['volatility']:.2f}%",
                    delta=f"{pstats['volatility'] - mstats['volatility']:+.2f}% vs market",
                    delta_color="inverse"
                )
            with sc3:
                st.metric(
                    "Sharpe Ratio",
                    f"{pstats['sharpe_ratio']:.3f}",
                    delta=f"{pstats['sharpe_ratio'] - mstats['sharpe_ratio']:+.3f} vs market"
                )

            st.caption(
                f"Market baseline — Return: {mstats['expected_return']:.2f}%, "
                f"Vol: {mstats['volatility']:.2f}%, Sharpe: {mstats['sharpe_ratio']:.3f}"
            )

# --- TAB 3: DEEP REPORT (Quant Reporter) ---
with tab_report:
    st.markdown("### Professional Quant Report")
    if not QUANT_REPORTER_AVAILABLE:
        st.error("`quant-reporter` library not installed. Please install it to use this feature.")
    else:
        st.write("Generate a comprehensive PDF-style HTML report with Backtesting, Efficient Frontier, and Monte Carlo simulations.")
        
        with st.expander("Report Configuration", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, SPY, QQQ")
                benchmark = st.text_input("Benchmark Ticker", "SPY")
            with col_in2:
                # Dates
                default_start = datetime.now() - timedelta(days=365*2)
                start_date = st.date_input("Training Start Date", default_start)
                end_date = st.date_input("Training End Date", datetime.now() - timedelta(days=90)) # Leave 90 days for Test
                
        if st.button("Generate Deep Report"):
            r_tickers = [t.strip() for t in repo_tickers.split(",") if t.strip()]
            if not r_tickers:
                st.warning("Enter tickers.")
            else:
                weight = 1.0 / len(r_tickers)
                portfolio_dict = {t: weight for t in r_tickers}
                report_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "portfolio_report.html"
                )
                report_file = os.path.normpath(report_file)
                with st.spinner("Generating report... (this may take 30-60 seconds)"):
                    try:
                        qr.create_combined_report(
                            portfolio_dict=portfolio_dict,
                            benchmark_ticker=benchmark,
                            train_start=start_date.strftime('%Y-%m-%d'),
                            train_end=end_date.strftime('%Y-%m-%d'),
                            filename=report_file
                        )
                        if os.path.exists(report_file):
                            with open(report_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.success("Report Generated Successfully!")
                            st.download_button(
                                label="Download HTML Report",
                                data=html_content,
                                file_name="My_Quant_Report.html",
                                mime="text/html"
                            )
                            st.divider()
                            st.subheader("Report Preview")
                            components.html(html_content, height=800, scrolling=True)
                        else:
                            st.error(f"Report file not created at {report_file}. Check Django terminal for errors.")
                    except Exception as e:
                        import traceback
                        st.error(f"Report Generation Failed: {e}")
                        st.code(traceback.format_exc())

