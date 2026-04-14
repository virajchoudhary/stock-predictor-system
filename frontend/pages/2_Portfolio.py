import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import threading
import os
import sys
import html
import re
from io import StringIO
from datetime import datetime, timedelta
import streamlit.components.v1 as components

_FRONTEND_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if _FRONTEND_ROOT not in sys.path:
    sys.path.insert(0, _FRONTEND_ROOT)

from reporting_utils import resolve_report_universe
from components.sidebar import render_backtest_sidebar

# Import quant_reporter
try:
    import quant_reporter as qr
    import yfinance as yf

    try:
        _YF_CACHE_DIR = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "yfinance")
        )
        os.makedirs(_YF_CACHE_DIR, exist_ok=True)
        yf.set_tz_cache_location(_YF_CACHE_DIR)
    except Exception:
        pass

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
                        rolling_returns[name] = f"{val:.2%}"
                    except Exception:
                        rolling_returns[name] = "N/A"
                else:
                    rolling_returns[name] = "Insufficient data"
            return pd.DataFrame(list(rolling_returns.items()), columns=["Period", "Return"])

        qr.opt_core.calculate_rolling_returns = patched_calculate_rolling_returns

        # CRITICAL: also patch the reference inside combined_report module
        # because it imports the function directly into its own namespace
        try:
            import quant_reporter.combined_report as _qr_combined
            _qr_combined.calculate_rolling_returns = patched_calculate_rolling_returns
        except Exception:
            pass
    # ---------------------------------------------------------------------------------

    # Monkey-patch html_builder to avoid emoji print crash on Windows cp1252
    import quant_reporter.html_builder as _qr_hb
    _orig_generate = _qr_hb.generate_html_report

    def _safe_generate_html_report(sections, title="Quantitative Report", filename="report.html"):
        import io, sys
        _old_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        try:
            _orig_generate(sections, title=title, filename=filename)
        finally:
            sys.stdout = _old_stdout

    _qr_hb.generate_html_report = _safe_generate_html_report

    QUANT_REPORTER_AVAILABLE = True
except Exception:
    QUANT_REPORTER_AVAILABLE = False

# Configuration
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Portfolio - Stock Price Predictor", layout="wide")
target_date = render_backtest_sidebar()

st.title("Portfolio Manager")

def _render_allocation_charts(allocation):
    if not allocation:
        st.warning("No allocation data.")
        return
    display_allocation = {
        ticker: weight for ticker, weight in allocation.items()
        if float(weight) > 0.0005
    } or allocation
    col_chart, col_table = st.columns(2)
    with col_chart:
        fig = go.Figure(data=[go.Pie(
            labels=list(display_allocation.keys()),
            values=list(display_allocation.values()),
            hole=0.4
        )])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, width='stretch')
    with col_table:
        df = pd.DataFrame(list(display_allocation.items()), columns=["Ticker", "Weight"])
        df["Weight"] = df["Weight"].apply(lambda x: f"{x:.1%}")
        st.table(df.set_index("Ticker"))


def _render_allocation_bar_table(allocation):
    if not allocation:
        st.warning("No allocation data.")
        return

    df = pd.DataFrame(list(allocation.items()), columns=["Ticker", "Weight"])
    df = df.sort_values("Weight", ascending=False)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Ticker"],
            y=df["Weight"],
            marker_color="#00CC96",
            text=[f"{v:.2%}" for v in df["Weight"]],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Weight",
            margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')

    with col_table:
        table_df = df.copy()
        table_df["Weight"] = table_df["Weight"].map(lambda x: f"{x:.2%}")
        st.dataframe(table_df.set_index("Ticker"), width='stretch')


def _build_allocation_bar_figure(allocation, title="Risk-Based Allocation"):
    df = pd.DataFrame(list(allocation.items()), columns=["Ticker", "Weight"])
    df = df.sort_values("Weight", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Ticker"],
        y=df["Weight"],
        marker_color="#00CC96",
        text=[f"{v:.2%}" for v in df["Weight"]],
        textposition="outside",
        hovertemplate="Ticker=%{x}<br>Weight=%{y:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Portfolio Weight",
        margin=dict(t=50, b=30),
        showlegend=False,
    )
    return fig


def _allocation_table_html(allocation, weight_column="Weight"):
    table_df = pd.DataFrame(list(allocation.items()), columns=["Ticker", weight_column])
    table_df = table_df.sort_values(weight_column, ascending=False)
    table_df[weight_column] = table_df[weight_column].map(lambda x: f"{x:.2%}")
    return table_df.to_html(index=False, classes="metrics-table")


def _history_payload_to_frame(payload):
    if not payload:
        return pd.DataFrame()

    dates = payload.get("dates", [])
    series = payload.get("series", {})
    if not dates or not series:
        return pd.DataFrame()

    frame = pd.DataFrame(series, index=pd.to_datetime(dates))
    frame.index.name = "Date"
    return frame.sort_index()


def _first_non_null(*values):
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return None


def _coerce_numeric(value, as_fraction=False):
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
        if not normalized:
            return None
        has_percent = normalized.endswith("%")
        if has_percent:
            normalized = normalized[:-1].strip()
        try:
            number = float(normalized)
        except ValueError:
            return None
        if as_fraction and (has_percent or abs(number) > 1.0):
            return number / 100.0
        return number

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if as_fraction and abs(number) > 1.0:
        return number / 100.0
    return number


def _format_weight_fraction(value):
    weight = _coerce_numeric(value, as_fraction=True)
    if weight is None:
        return "N/A"
    return f"{weight:.2%}"


def _build_bl_return_frame(bl_res, tickers):
    raw_rows = {}
    for row in bl_res.get("return_table", []) or []:
        ticker = _first_non_null(row.get("ticker"), row.get("Ticker"))
        if ticker:
            raw_rows[str(ticker)] = row

    equilibrium_returns = bl_res.get("equilibrium_returns", {})
    posterior_returns = bl_res.get("posterior_returns", {})
    betas = bl_res.get("betas", {})
    equilibrium_weights = bl_res.get("equilibrium_weights", {})
    bl_weights = bl_res.get("bl_weights", {})

    normalized_rows = []
    ordered_tickers = tickers or list(raw_rows.keys()) or list(bl_weights.keys()) or list(equilibrium_weights.keys())
    for ticker in ordered_tickers:
        raw = raw_rows.get(ticker, {})

        market_weight = _coerce_numeric(
            _first_non_null(
                equilibrium_weights.get(ticker),
                raw.get("market_weight"),
                raw.get("Market Weight"),
                raw.get("equilibrium_weight"),
            ),
            as_fraction=True,
        )
        bl_weight = _coerce_numeric(
            _first_non_null(
                bl_weights.get(ticker),
                raw.get("bl_weight"),
                raw.get("BL Weight"),
                raw.get("optimal_weight"),
                raw.get("Optimal Weight"),
            ),
            as_fraction=True,
        )
        weight_tilt = _coerce_numeric(
            _first_non_null(raw.get("weight_tilt"), raw.get("Weight Tilt")),
            as_fraction=True,
        )

        if market_weight is None:
            market_weight = 0.0
        if bl_weight is None:
            bl_weight = 0.0
        if weight_tilt is None:
            weight_tilt = bl_weight - market_weight

        normalized_rows.append({
            "ticker": ticker,
            "beta_vs_benchmark": _coerce_numeric(
                _first_non_null(
                    betas.get(ticker),
                    raw.get("beta_vs_benchmark"),
                    raw.get("Beta vs S&P 500"),
                    raw.get("Beta"),
                )
            ),
            "equilibrium_return_pct": _coerce_numeric(
                _first_non_null(
                    equilibrium_returns.get(ticker),
                    raw.get("equilibrium_return_pct"),
                    raw.get("pi_return_pct"),
                    raw.get("Pi (Eq Return %)"),
                    raw.get("Equilibrium Return %"),
                )
            ),
            "posterior_return_pct": _coerce_numeric(
                _first_non_null(
                    posterior_returns.get(ticker),
                    raw.get("posterior_return_pct"),
                    raw.get("bl_return_pct"),
                    raw.get("BL Return %"),
                    raw.get("Posterior Return %"),
                )
            ),
            "market_weight": market_weight,
            "bl_weight": bl_weight,
            "weight_tilt": round(float(weight_tilt), 4),
        })

    return pd.DataFrame(normalized_rows)


def _normalize_weight_vector(weight_dict, tickers):
    weights = pd.Series(weight_dict, dtype=float).reindex(tickers).fillna(0.0)
    total = float(weights.sum())
    if total <= 1e-10:
        return pd.Series({t: 1.0 / len(tickers) for t in tickers})
    return weights / total


def _portfolio_growth(price_frame, weight_dict):
    if price_frame.empty:
        return pd.Series(dtype=float)

    tickers = list(price_frame.columns)
    weights = _normalize_weight_vector(weight_dict, tickers)
    normalized = price_frame / price_frame.iloc[0]
    growth = normalized.mul(weights, axis=1).sum(axis=1)
    growth.name = "Growth"
    return growth


def _performance_metrics(growth_series):
    if growth_series.empty or len(growth_series) < 2:
        return None

    total_return = float(growth_series.iloc[-1] / growth_series.iloc[0] - 1.0)
    daily_returns = growth_series.pct_change().dropna()
    if daily_returns.empty:
        return None

    annualized_return = float((1.0 + total_return) ** (252.0 / len(daily_returns)) - 1.0)
    running_peak = growth_series.cummax()
    max_drawdown = float(((growth_series - running_peak) / running_peak).min())
    pnl_pct = total_return * 100.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "pnl_pct": pnl_pct,
    }


def _asset_window_returns(price_frame):
    if price_frame.empty or len(price_frame) < 2:
        return {}
    returns = (price_frame.iloc[-1] / price_frame.iloc[0] - 1.0) * 100.0
    return {
        ticker: round(float(value), 2)
        for ticker, value in returns.items()
    }


def _window_return_percent(price_series):
    if price_series is None or len(price_series) < 2:
        return None
    return round(float((price_series.iloc[-1] / price_series.iloc[0] - 1.0) * 100.0), 2)


def _figure_has_data(fig):
    if fig is None or not getattr(fig, "data", None):
        return False

    for trace in fig.data:
        for attr in ("x", "y", "z", "values"):
            values = getattr(trace, attr, None)
            if values is not None and len(values) > 0:
                return True
    return False


def _empty_report_figure(title, message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="white"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=80, r=40, t=60, b=60),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def _flatten_table_df(df):
    """Flatten MultiIndex columns, strip unnamed levels, ensure clean single-level names."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col_tuple in df.columns:
            parts = [str(c).strip() for c in col_tuple
                     if c is not None and not re.match(r'^Unnamed', str(c).strip())]
            if parts:
                new_cols.append(" ".join(parts))
            else:
                new_cols.append(f"Col_{len(new_cols)}")
        df.columns = new_cols
    else:
        clean_cols = []
        for c in df.columns:
            name = str(c).strip()
            if re.match(r'^Unnamed', name):
                name = ""
            clean_cols.append(name)
        df.columns = clean_cols
    # Also rename first column if it is empty or blank to something sensible
    if df.columns[0] == "" or df.columns[0].startswith("Col_"):
        cols = list(df.columns)
        cols[0] = "Metric"
        df.columns = cols
    return df


def _style_report_figure(fig, fallback_title, xaxis_title=None, yaxis_title=None, height=400):
    if fig is None:
        return None

    title_obj = fig.layout.title.text if getattr(fig.layout, "title", None) else None
    height = max(height, 400)  # minimum 400px
    fig.update_layout(
        title=title_obj or fallback_title,
        height=height,
        margin=dict(l=60, r=30, t=40, b=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    has_cartesian_axes = any(getattr(trace, "type", None) not in {"pie"} for trace in fig.data)
    if has_cartesian_axes:
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.1)",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.1)",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
        )
        if xaxis_title:
            fig.update_xaxes(title_text=xaxis_title)
        elif not fig.layout.xaxis.title.text:
            fig.update_xaxes(title_text="Date")

        if yaxis_title:
            fig.update_yaxes(title_text=yaxis_title)

    return fig


def _prepare_report_figure(fig, title, xaxis_title=None, yaxis_title=None, height=400):
    if fig is None or not _figure_has_data(fig):
        return _empty_report_figure(title, f"{title} is not available for the current data selection.")
    return _style_report_figure(
        fig,
        fallback_title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=max(height, 400),
    )


def _validate_benchmark_ticker(benchmark_ticker, start_date, end_date):
    ticker = benchmark_ticker.strip().upper()
    if not ticker:
        return None, "Benchmark ticker is required."

    try:
        history = yf.download(
            ticker,
            start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            end=(pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            threads=False,
        )
        if history is None or history.empty:
            return None, f"`{ticker}` did not return benchmark price history for the selected dates."
    except Exception as exc:
        return None, f"Could not validate benchmark ticker `{ticker}`: {exc}"

    return ticker, None


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_sector_map(tickers):
    sector_map = {}
    for ticker in tickers:
        sector = "Other"
        try:
            info = yf.Ticker(ticker).info if QUANT_REPORTER_AVAILABLE else {}
            sector = (
                info.get("sector")
                or info.get("industry")
                or info.get("quoteType")
                or "Other"
            )
        except Exception:
            sector = "Other"
        sector_map[ticker] = str(sector)
    return sector_map


def _response_error_message(response):
    try:
        payload = response.json()
    except Exception:
        payload = {}
    return payload.get("error") or f"Request returned {response.status_code}"


def _request_bl_allocation(tickers, risk_tolerance, target_date=None):
    try:
        response = requests.post(
            f"{API_URL}/bl-optimize/",
            json={"tickers": tickers, "risk_tolerance": risk_tolerance, "target_date": target_date},
            timeout=45,
        )
        if response.status_code != 200:
            return None, _response_error_message(response)
        return response.json(), None
    except requests.Timeout:
        return None, "The Black-Litterman allocation service timed out."
    except Exception as exc:
        return None, str(exc)


def _request_risk_allocation(tickers, risk_tolerance, target_date=None):
    try:
        response = requests.post(
            f"{API_URL}/optimize/",
            json={"tickers": tickers, "risk_tolerance": risk_tolerance, "target_date": target_date},
            timeout=20,
        )
        if response.status_code != 200:
            return None, _response_error_message(response)
        return response.json(), None
    except requests.Timeout:
        return None, "The risk-based allocation service timed out."
    except Exception as exc:
        return None, str(exc)


def _fetch_auto_allocation(tickers, risk_tolerance, target_date=None):
    bl_result, bl_error = _request_bl_allocation(tickers, risk_tolerance, target_date=target_date)
    if bl_result and bl_result.get("allocation"):
        bl_result["method"] = "bl"
        bl_result["bl_unavailable_reason"] = None
        return bl_result, None

    risk_result, risk_error = _request_risk_allocation(tickers, risk_tolerance, target_date=target_date)
    if risk_result and risk_result.get("allocation"):
        risk_result["method"] = "standard"
        risk_result["bl_unavailable_reason"] = bl_error
        return risk_result, None

    return None, risk_error or bl_error or "Allocation services were unavailable."


def _format_duration(seconds):
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _render_deep_report_progress(
    placeholder,
    progress_value,
    stage_label,
    detail,
    start_time,
    eta_seconds=None,
):
    progress_value = max(0.0, min(float(progress_value), 1.0))
    progress_percent = int(round(progress_value * 100))
    timer_id = f"deep-report-elapsed-{int(start_time * 1000)}"
    eta_id = f"deep-report-eta-{int(start_time * 1000)}"
    stage_label_safe = html.escape(stage_label)
    detail_safe = html.escape(detail)
    is_complete = progress_value >= 1.0
    elapsed_now = time.time() - start_time

    with placeholder.container():
        st.progress(progress_percent)
        st.caption(f"{stage_label} ({progress_percent}%)")
        components.html(
            f"""
            <div style="border:1px solid #1f2937;border-radius:12px;padding:14px 16px;background:#0f172a;color:#e2e8f0;font-family:Inter,Segoe UI,Arial,sans-serif;">
              <div style="display:flex;gap:16px;justify-content:space-between;flex-wrap:wrap;">
                <div style="min-width:160px;">
                  <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Stage</div>
                  <div style="font-size:16px;font-weight:600;margin-top:4px;">{stage_label_safe}</div>
                </div>
                <div style="min-width:120px;">
                  <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Elapsed</div>
                  <div id="{timer_id}" style="font-size:16px;font-weight:600;margin-top:4px;">0s</div>
                </div>
                <div style="min-width:180px;">
                  <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Estimated Remaining</div>
                  <div id="{eta_id}" style="font-size:16px;font-weight:600;margin-top:4px;">Calculating...</div>
                </div>
              </div>
              <div style="font-size:13px;color:#cbd5e1;margin-top:12px;">{detail_safe}</div>
            </div>
            <script>
            (function() {{
              const startTime = {int(start_time * 1000)};
              const renderTime = Date.now();
              const progressNow = {progress_value};
              const elapsedAtRender = {elapsed_now};
              const etaAtRender = {eta_seconds if eta_seconds is not None else -1};
              const isComplete = {'true' if is_complete else 'false'};
              const timerEl = document.getElementById("{timer_id}");
              const etaEl = document.getElementById("{eta_id}");

              function formatDuration(totalSeconds) {{
                totalSeconds = Math.max(0, Math.floor(totalSeconds));
                const hours = Math.floor(totalSeconds / 3600);
                const minutes = Math.floor((totalSeconds % 3600) / 60);
                const seconds = totalSeconds % 60;
                if (hours > 0) return hours + "h " + minutes + "m " + seconds + "s";
                if (minutes > 0) return minutes + "m " + seconds + "s";
                return seconds + "s";
              }}

              function update() {{
                if (!timerEl) return;
                const now = Date.now();
                if (isComplete) {{
                  // Freeze the elapsed timer at the moment the report completed
                  timerEl.textContent = formatDuration(elapsedAtRender);
                  if (etaEl) etaEl.textContent = "Done!";
                  return;
                }}
                const elapsedSeconds = Math.max(0, (now - startTime) / 1000);
                timerEl.textContent = formatDuration(elapsedSeconds);

                if (etaEl) {{
                  if (etaAtRender > 0) {{
                    // Compute how many seconds have passed since this render
                    const sinceRender = Math.max(0, (now - renderTime) / 1000);
                    const remaining = Math.max(0, etaAtRender - sinceRender);
                    etaEl.textContent = remaining < 1 ? "Almost done..." : formatDuration(remaining);
                  }} else {{
                    etaEl.textContent = "Calculating...";
                  }}
                }}
              }}

              update();
              if (!isComplete) {{
                window.setInterval(update, 1000);
              }}
            }})();
            </script>
            """,
            height=118,
        )
        st.caption("Progress is stage-based. Runtime usually depends on ticker count and date range.")


def _build_deep_report_context(
    input_tickers,
    allocation_result,
    benchmark_ticker,
    train_start,
    train_end,
    report_file,
    sector_map,
    risk_tolerance,
    progress_callback=None,
):
    from quant_reporter.data import get_data
    from quant_reporter.metrics import calculate_metrics
    from quant_reporter.html_builder import generate_html_report
    from quant_reporter.plotting import plot_cumulative_returns, plot_regression
    from quant_reporter.opt_core import calculate_rolling_returns, get_portfolio_price
    import quant_reporter.combined_report as qr_combined

    def _emit_progress(progress_value, stage_label, detail, eta_seconds=None):
        if progress_callback:
            progress_callback(progress_value, stage_label, detail, eta_seconds)

    requested_tickers = [str(t).upper() for t in input_tickers]
    allocation_weights = allocation_result.get("allocation", allocation_result)
    allocation_source = allocation_result.get("source", "risk_based")
    allocation_blend_meta = allocation_result.get("blend_meta", {})
    allocation_fallback_reason = (
        allocation_result.get("fallback_reason")
        or allocation_result.get("bl_unavailable_reason")
    )
    allocation_source_label = (
        "AI-Driven Black-Litterman"
        if allocation_source == "bl"
        else "Risk-Based Fallback"
    )

    # ── Bug 3: Remove benchmark from investable asset universe ─────────
    benchmark_excluded = False
    if benchmark_ticker.upper() in [t.upper() for t in requested_tickers]:
        requested_tickers = [t for t in requested_tickers if t.upper() != benchmark_ticker.upper()]
        allocation_weights = {k: v for k, v in allocation_weights.items() if k.upper() != benchmark_ticker.upper()}
        benchmark_excluded = True
    # ──────────────────────────────────────────────────────────────────────

    test_start_dt = pd.to_datetime(train_end) + timedelta(days=1)
    test_end_dt = datetime.now() - timedelta(days=1)
    if test_start_dt >= test_end_dt:
        raise ValueError("Training end date must leave at least one day for the test period.")

    train_start_str = pd.to_datetime(train_start).strftime("%Y-%m-%d")
    train_end_str = pd.to_datetime(train_end).strftime("%Y-%m-%d")
    test_start_str = test_start_dt.strftime("%Y-%m-%d")
    test_end_str = test_end_dt.strftime("%Y-%m-%d")
    full_end_str = test_end_str

    _emit_progress(0.28, "Preparing report inputs", "Loading report settings and risk-free rate.", 55)
    _indian_count = sum(1 for t in requested_tickers if t.upper().endswith((".NS", ".BO")))
    if _indian_count > len(requested_tickers) / 2:
        risk_free_rate = 0.065  # RBI repo rate for Indian market
    else:
        risk_free_rate = 0.0359
    all_tickers = list(dict.fromkeys(requested_tickers + [benchmark_ticker]))

    _emit_progress(0.38, "Fetching full-period data", f"Downloading price history through {full_end_str}.", 45)
    data_full = get_data(all_tickers, train_start_str, full_end_str)
    _emit_progress(0.48, "Fetching training data", f"Loading in-sample data through {train_end_str}.", 34)
    data_train = get_data(all_tickers, train_start_str, train_end_str)
    _emit_progress(0.58, "Fetching validation data", f"Loading out-of-sample data through {test_end_str}.", 26)
    data_test = get_data(all_tickers, test_start_str, test_end_str)
    if data_full is None or data_train is None or data_test is None:
        raise ValueError("Failed to fetch report data for one or more periods.")

    report_universe = resolve_report_universe(
        requested_tickers,
        allocation_weights,
        data_full,
        data_train,
        data_test,
    )
    tickers = report_universe["analysis_tickers"]
    portfolio_dict = report_universe["allocation_weights"]
    filtered_sector_map = {ticker: sector_map.get(ticker, "Other") for ticker in tickers}

    if len(tickers) < 2:
        raise ValueError(
            "Need at least 2 requested tickers with shared history across the training and test windows."
        )

    _emit_progress(0.68, "Calculating portfolio metrics", "Building cumulative returns, regression, and rolling metrics.", 19)
    portfolio_eval = (data_full[[benchmark_ticker]] / data_full[[benchmark_ticker]].iloc[0]).copy()
    portfolio_eval["My Portfolio"] = get_portfolio_price(data_full[tickers], portfolio_dict)
    pr_metrics, pr_plot_data = calculate_metrics(
        portfolio_eval, "My Portfolio", benchmark_ticker, risk_free_rate
    )

    per_asset_swot = {}
    data_1y = data_full.iloc[-252:] if len(data_full) > 252 else data_full
    for ticker in tickers:
        try:
            ret_1y = (data_1y[ticker].iloc[-1] / data_1y[ticker].iloc[0]) - 1
            t_eval = pd.DataFrame({benchmark_ticker: portfolio_eval[benchmark_ticker], ticker: data_full[ticker] / data_full[ticker].iloc[0]})
            t_met, _ = calculate_metrics(t_eval, ticker, benchmark_ticker, risk_free_rate)
            t_ret = data_full[ticker].pct_change().dropna()
            p_ret = portfolio_eval["My Portfolio"].pct_change().dropna()
            b_ret = data_full[benchmark_ticker].pct_change().dropna()
            corr_port = pd.concat([t_ret, p_ret], axis=1).dropna().corr().iloc[0, 1]
            corr_bench = pd.concat([t_ret, b_ret], axis=1).dropna().corr().iloc[0, 1]
            def _to_float(val, default=0.0):
                try:
                    return float(str(val).replace("%", "").replace(",", "").strip()) / 100 if "%" in str(val) else float(val)
                except (ValueError, TypeError):
                    return default

            per_asset_swot[ticker] = {
                "cagr_1y": float(ret_1y),
                "beta": _to_float(t_met.get("Beta (vs Benchmark)", 1.0)),
                "volatility": _to_float(t_met.get("Annualized Volatility (Asset)", 0.2)),
                "max_drawdown": _to_float(t_met.get("Max Drawdown", 0)),
                "corr_port": float(corr_port),
                "corr_bench": float(corr_bench),
                "weight": float(portfolio_dict.get(ticker, 0)),
                "sector": filtered_sector_map.get(ticker, "Other"),
            }
        except Exception:
            pass

    # ── Bug 4: Build rolling returns as clean DataFrame ────────────────
    try:
        pr_rolling_df = calculate_rolling_returns(portfolio_eval)
        # The patched version returns [Period, Return] columns directly
        if isinstance(pr_rolling_df, pd.DataFrame) and "Period" in pr_rolling_df.columns:
            pr_rolling_html = pr_rolling_df.to_html(index=False, classes="metrics-table")
        else:
            pr_rolling_html = pr_rolling_df.to_html(classes="metrics-table")
    except Exception:
        pr_rolling_html = "<table><tr><td>Rolling returns unavailable</td></tr></table>"
    # ──────────────────────────────────────────────────────────────────────

    pr_plots = {
        "cumulative": plot_cumulative_returns(pr_plot_data, "My Portfolio", benchmark_ticker),
        "regression": plot_regression(pr_plot_data, pr_metrics, "My Portfolio", benchmark_ticker),
    }

    _emit_progress(0.82, "Running backtests and simulations", "Executing optimization validation and Monte Carlo analysis.", 10)
    validation = qr_combined._run_validation_logic(
        data_train,
        data_test,
        tickers,
        tickers,
        benchmark_ticker,
        portfolio_dict,
        risk_free_rate,
        filtered_sector_map,
        None,
        None,
        filtered_sector_map,
    )

    # ── Bug 2 & 6: Fix validation table — correct Sharpe scaling & clean column names ──
    try:
        val_table_dfs = pd.read_html(StringIO(validation["table_html"]))
        if val_table_dfs:
            val_df = _flatten_table_df(val_table_dfs[0])
            # The table from combined_report has "Portfolio" as first col, then metrics
            # We need to identify columns and reformat
            col_rename = {}
            for c in val_df.columns:
                cl = c.lower()
                if 'portfolio' in cl or c == 'Metric':
                    col_rename[c] = 'Portfolio'
                elif 'in-sample' in cl and 'cagr' in cl:
                    col_rename[c] = 'IS CAGR'
                elif 'out-of-sample' in cl and 'cagr' in cl:
                    col_rename[c] = 'OOS CAGR'
                elif 'in-sample' in cl and 'volatil' in cl:
                    col_rename[c] = 'IS Vol'
                elif 'out-of-sample' in cl and 'volatil' in cl:
                    col_rename[c] = 'OOS Vol'
                elif 'in-sample' in cl and 'sharpe' in cl:
                    col_rename[c] = 'IS Sharpe'
                elif 'out-of-sample' in cl and 'sharpe' in cl:
                    col_rename[c] = 'OOS Sharpe'
                elif 'in-sample' in cl and ('drawdown' in cl or 'max' in cl):
                    col_rename[c] = 'IS MaxDD'
                elif 'out-of-sample' in cl and ('drawdown' in cl or 'max' in cl):
                    col_rename[c] = 'OOS MaxDD'
                elif 'in-sample' in cl and 'alpha' in cl:
                    col_rename[c] = 'IS Alpha'
                elif 'out-of-sample' in cl and 'alpha' in cl:
                    col_rename[c] = 'OOS Alpha'
            if col_rename:
                val_df = val_df.rename(columns=col_rename)

            # Bug 2: Fix Sharpe/Sortino/Calmar values that were formatted as % then
            # converted back incorrectly. Sharpe values should be around -3..+3.
            # The combined_report pipeline stores Sharpe as "1.23" string, then
            # the to_html with {:.2%}.format multiplies by 100 — causing 123%.
            # We re-parse and fix ratio columns (non-percentage metrics).
            for ratio_col in ['IS Sharpe', 'OOS Sharpe']:
                if ratio_col in val_df.columns:
                    def _fix_ratio(v):
                        try:
                            s = str(v).strip().replace('%', '').replace(',', '')
                            num = float(s)
                            # If the value looks like it was multiplied by 100 (e.g. -232 instead of -2.32)
                            if abs(num) > 10:
                                num = num / 100.0
                            return f"{num:.2f}"
                        except (ValueError, TypeError):
                            return v
                    val_df[ratio_col] = val_df[ratio_col].apply(_fix_ratio)

            validation["table_html"] = val_df.to_html(index=False, classes="metrics-table")
    except Exception:
        pass
    # ──────────────────────────────────────────────────────────────────────

    allocation_method = (
        "AI-Driven Black-Litterman"
        if allocation_source == "bl"
        else (
            f"{allocation_blend_meta.get('from', 'Inverse Vol')} -> "
            f"{allocation_blend_meta.get('to', 'Max Sharpe')} "
            f"(alpha {allocation_blend_meta.get('alpha', 0):.2f})"
            if allocation_blend_meta
            else "Continuous blend: inverse vol -> min vol -> max Sharpe"
        )
    )

    allocation_metrics = {
        "Risk Tolerance": f"{risk_tolerance:.2f}",
        "Allocation Source": allocation_source_label,
        "Allocation Method": allocation_method,
        "Benchmark": benchmark_ticker,
        "Report Universe": ", ".join(tickers),
    }
    if benchmark_excluded:
        allocation_metrics["Benchmark Exclusion"] = f"{benchmark_ticker} excluded from portfolio optimization — it is used as benchmark"
    if report_universe["dropped_tickers"]:
        allocation_metrics["Unavailable Tickers"] = ", ".join(report_universe["dropped_tickers"])
    if allocation_fallback_reason:
        allocation_metrics["Fallback Reason"] = allocation_fallback_reason

    allocation_section = {
        "title": "Portfolio Construction",
        "description": (
            f"{allocation_source_label} allocation for the requested report universe "
            f"using benchmark {benchmark_ticker}."
            + (f" Note: {benchmark_ticker} excluded from portfolio optimization — it is used as benchmark." if benchmark_excluded else "")
        ),
        "sidebar": [
            {
                "title": "Allocation Inputs",
                "type": "metrics",
                "data": allocation_metrics,
            },
        ],
        "main_content": [
            {
                "title": "Portfolio Allocation Bar Chart",
                "type": "plot",
                "data": _build_allocation_bar_figure(portfolio_dict, "Portfolio Allocation"),
            },
            {
                "title": "Portfolio Allocation Table",
                "type": "table_html",
                "data": _allocation_table_html(portfolio_dict, "Weight"),
            },
        ],
    }

    # Accuracy Improvement: Portfolio Backtest Verification
    # Compare generated portfolio vs benchmark over the subsequent 30 days
    if target_date:
        try:
            from .reporting_utils import get_portfolio_price
            from api.ai_services import get_price_history
            
            # Use test data (OOS) which covers the period after target_date
            # Or fetch specifically for the 30 days after target_date
            bench_after = data_full[benchmark_ticker].copy()
            port_after = get_portfolio_price(data_full[tickers], portfolio_dict)
            
            # Normalize to 1.0 at target_date (iloc[-test_len] or similar)
            # Actually, target_date is the cutoff. So data_test is what matters.
            if not data_test.empty:
                b_ret_oos = (data_test[benchmark_ticker].iloc[-1] / data_test[benchmark_ticker].iloc[0]) - 1
                p_ret_oos = (get_portfolio_price(data_test[tickers], portfolio_dict).iloc[-1] / 
                             get_portfolio_price(data_test[tickers], portfolio_dict).iloc[0]) - 1
                
                allocation_section["sidebar"].append({
                    "title": "Backtest Verification (OOS)",
                    "type": "metrics",
                    "data": {
                        "Portfolio Return": f"{p_ret_oos:+.2%}",
                        "Benchmark Return": f"{b_ret_oos:+.2%}",
                        "Alpha Extraction": f"{(p_ret_oos - b_ret_oos):+.2%}",
                        "Verdict": "SUCCESS" if p_ret_oos > b_ret_oos else "MARKET ALPHA GAP"
                    }
                })
        except Exception:
            pass

    sections = [
        allocation_section,
        {
            "title": "User Portfolio",
            "description": f"Full-period analysis from {train_start_str} to {full_end_str}.",
            "sidebar": [
                {"title": "User Portfolio Metrics", "type": "metrics", "data": pr_metrics},
                {"title": "User Portfolio Rolling Returns", "type": "table_html", "data": pr_rolling_html},
            ],
            "main_content": [
                {"title": "Portfolio Cumulative Returns", "type": "plot", "data": pr_plots["cumulative"]},
                {"title": "Portfolio Alpha/Beta Regression", "type": "plot", "data": pr_plots["regression"]},
            ],
        },
        {
            "title": "Optimization Analysis",
            "description": f"Training-period optimization from {train_start_str} to {train_end_str}.",
            "sidebar": [
                {"title": "Asset-Benchmark Correlation", "type": "table_html", "data": validation["asset_corr_html"]},
            ],
            "main_content": [
                {"title": "Strategy Compositions (Asset)", "type": "plot", "data": validation["optimization_plots"]["pie_plot"]},
                {"title": "Strategy Compositions (Sector)", "type": "plot", "data": validation["optimization_plots"]["sector_pie_plot"]},
                {"title": "Risk Contribution (Asset)", "type": "plot", "data": validation["optimization_plots"]["risk_contribution"]},
                {"title": "Risk Contribution (Sector)", "type": "plot", "data": validation["optimization_plots"]["sector_risk_contribution"]},
                {"title": "Rolling Sharpe", "type": "plot", "data": validation["optimization_plots"]["rolling_sharpe_plot"]},
                {"title": "Efficient Frontier", "type": "plot", "data": validation["optimization_plots"]["frontier"]},
                {"title": "Correlation Heatmap", "type": "plot", "data": validation["optimization_plots"]["heatmap"]},
            ],
        },
        {
            "title": "Walk-Forward Validation",
            "description": f"Out-of-sample performance from {test_start_str} to {test_end_str}.",
            "sidebar": [],
            "main_content": [
                {"title": "In-Sample vs Out-of-Sample Performance", "type": "table_html", "data": validation["table_html"]},
                {"title": "Out-of-Sample Cumulative Returns", "type": "plot", "data": validation["validation_plots"]["cumulative_plot"]},
                {"title": "Out-of-Sample Drawdown", "type": "plot", "data": validation["validation_plots"]["drawdown_plot"]},
            ],
        },
        {
            "title": "Monte Carlo Simulation",
            "description": f"Simulation horizon matched to {validation['test_days']} trading days.",
            "sidebar": [
                {"title": "Simulation Risk Metrics", "type": "metrics", "data": validation["mc_metrics"]},
                {"title": "Success Probabilities", "type": "metrics", "data": validation["mc_probs"]},
            ],
            "main_content": [
                {"title": "Projected Future Paths", "type": "plot", "data": validation["mc_plots"]["paths"]},
                {"title": "Distribution of Final Returns", "type": "plot", "data": validation["mc_plots"]["dist"]},
                {"title": "Probability of Exceeding Return", "type": "plot", "data": validation["mc_plots"]["prob_curve"]},
            ],
        },
        {
            "title": "SWOT Analysis",
            "description": "Strengths, Weaknesses, Opportunities, and Threats of the portfolio composition.",
            "sidebar": [],
            "main_content": [
                {"title": "SWOT Summary", "type": "swot_placeholder", "data": {
                    "cagr_asset": pr_metrics.get("CAGR (Asset)"),
                    "cagr_bench": pr_metrics.get("CAGR (Benchmark)"),
                    "sharpe": pr_metrics.get("Sharpe Ratio (Asset)"),
                    "sortino": pr_metrics.get("Sortino Ratio (Asset)"),
                    "calmar": pr_metrics.get("Calmar Ratio (Asset)"),
                    "beta": pr_metrics.get("Beta (vs Benchmark)"),
                    "alpha": pr_metrics.get("Alpha (Annualized)"),
                    "var_95": pr_metrics.get("VaR (95%)"),
                    "max_drawdown": pr_metrics.get("Max Drawdown"),
                    "volatility_asset": pr_metrics.get("Annualized Volatility (Asset)"),
                    "volatility_bench": pr_metrics.get("Annualized Volatility (Bench)"),
                    "tickers": tickers,
                    "allocation_weights": portfolio_dict,
                    "benchmark": benchmark_ticker,
                    "sector_map": filtered_sector_map,
                    "benchmark_excluded": benchmark_excluded,
                    "asset_corr_html": validation.get("asset_corr_html", ""),
                    "per_asset_swot": per_asset_swot,
                }},
            ],
        },
    ]

    axis_title_overrides = {
        "Portfolio Allocation Bar Chart": ("Ticker", "Portfolio Weight"),
        "Portfolio Cumulative Returns": ("Date", "Growth of $1"),
        "Portfolio Alpha/Beta Regression": (benchmark_ticker, "Portfolio Return"),
        "Strategy Compositions (Asset)": (None, None),
        "Strategy Compositions (Sector)": (None, None),
        "Risk Contribution (Asset)": ("Portfolio", "Percent of Total Risk"),
        "Risk Contribution (Sector)": ("Portfolio", "Percent of Total Risk"),
        "Rolling Sharpe": ("Date", "Sharpe Ratio"),
        "Efficient Frontier": ("Annualized Volatility (Risk)", "Annualized Return"),
        "Correlation Heatmap": (None, None),
        "Out-of-Sample Cumulative Returns": ("Date", "Growth of $1"),
        "Out-of-Sample Drawdown": ("Date", "Drawdown"),
        "Projected Future Paths": ("Trading Day", "Growth of $1"),
        "Distribution of Final Returns": ("Final Return", "Frequency"),
        "Probability of Exceeding Return": ("Return Threshold", "Probability"),
    }

    _emit_progress(0.91, "Formatting charts", "Preparing figures and tables for the dashboard view.", 5)
    for section in sections:
        for block in section.get("main_content", []):
            if block.get("type") != "plot":
                continue
            xaxis_title, yaxis_title = axis_title_overrides.get(block["title"], (None, None))
            # Bug 10: Regression chart needs minimum 400px
            h = 400
            if "Efficient Frontier" in block["title"] or "Projected Future Paths" in block["title"]:
                h = 450
            block["data"] = _prepare_report_figure(
                block["data"],
                title=block["title"],
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                height=h,
            )

    _emit_progress(0.97, "Building HTML report", "Generating the downloadable report file.", 2)
    generate_html_report(sections, title="Combined Portfolio Report", filename=report_file)
    with open(report_file, "r", encoding="utf-8") as file_handle:
        html_content = file_handle.read()

    _emit_progress(1.0, "Report complete", "Dashboard and HTML export are ready.", 0)
    return {
        "html_content": html_content,
        "risk_free_rate": risk_free_rate,
        "sections": sections,
        "allocation_result": allocation_result,
        "report_universe": report_universe,
        "benchmark_excluded": benchmark_excluded,
    }


def _render_swot_section(swot_data):
    """Render SWOT cards from computed portfolio metrics. Bullet-proof with error display."""
    try:
        tickers = swot_data.get("tickers", [])
        benchmark = swot_data.get("benchmark", "SPY")
        portfolio_dict = swot_data.get("allocation_weights", {})
        sector_map = swot_data.get("sector_map", {})
        benchmark_excluded = swot_data.get("benchmark_excluded", False)
        asset_corr_html = swot_data.get("asset_corr_html", "")

        _is_indian = any(t.upper().endswith((".NS", ".BO")) for t in tickers)
        _market_listing = "NSE-listed equities" if _is_indian else "US-listed equities"
        _market_name = "Indian markets" if _is_indian else "US markets"

        # ── Safe numeric parsers ───────────────────────────────────────
        def _pct(key):
            """Parse '18.04%' → 0.1804, return None on failure."""
            v = swot_data.get(key)
            if v is None:
                return None
            try:
                return float(str(v).replace("%", "").replace(",", "").strip()) / 100
            except Exception:
                return None

        def _flt(key):
            """Parse '1.32' → 1.32, return None on failure."""
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

        # ── Parse correlation table ────────────────────────────────────
        corr_pairs = {}
        if asset_corr_html:
            try:
                from io import StringIO as _SIO
                cdf = pd.read_html(_SIO(asset_corr_html))[0]
                for _, row in cdf.iterrows():
                    try:
                        corr_pairs[str(row.iloc[0])] = float(row.iloc[-1])
                    except Exception:
                        pass
            except Exception:
                pass

        high_corr = {t: v for t, v in corr_pairs.items() if v > 0.8}

        # ── Composition analysis ───────────────────────────────────────
        sectors = list(set(sector_map.values())) if sector_map else []
        sector_wts = {}
        for t, w in portfolio_dict.items():
            sector_wts[sector_map.get(t, "Other")] = sector_wts.get(sector_map.get(t, "Other"), 0) + w
        dom_sector = max(sector_wts.items(), key=lambda x: x[1]) if sector_wts else ("N/A", 0)
        top3 = sorted(portfolio_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        max_wt = max(portfolio_dict.values()) if portfolio_dict else 0
        tech_tickers = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMD", "QQQ", "AVGO", "TSM"}
        has_ai = bool(set(t.upper() for t in tickers) & {"NVDA", "QQQ"})
        tech_alloc = sum(w for t, w in portfolio_dict.items() if t.upper() in tech_tickers)

        # ════════════════════════════════════════════════════════════════
        # STRENGTHS (green)
        # ════════════════════════════════════════════════════════════════
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
            S.append(f"Calmar ratio of {calmar:.2f} shows returns adequately compensate for drawdown risk.")
        # Guarantee ≥ 3
        if len(S) < 3 and top3:
            S.append(f"Top holdings: {', '.join(f'{t} ({w:.1%})' for t,w in top3)}.")
        if len(S) < 3 and benchmark_excluded:
            S.append(f"Benchmark ({benchmark}) correctly excluded from the investable universe.")
        _s_fallbacks = [
            f"Portfolio consists of {len(tickers)} liquid, {_market_listing} with deep historical data.",
            f"All holdings are exchange-traded with transparent daily pricing in {_market_name}.",
            f"Portfolio benefits from broad institutional coverage across {len(tickers)} names.",
        ]
        for _fb in _s_fallbacks:
            if len(S) >= 3:
                break
            S.append(_fb)

        # ════════════════════════════════════════════════════════════════
        # WEAKNESSES (amber)
        # ════════════════════════════════════════════════════════════════
        W = []
        if beta is not None and beta > 1.2:
            W.append(f"High market sensitivity — beta of {beta:.2f} indicates {(beta-1)*100:.0f}% more volatility than the benchmark.")
        if max_dd is not None and max_dd < -0.20:
            W.append(f"Significant drawdown risk of {max_dd:.2%} — portfolio experienced substantial peak-to-trough decline.")
        if alpha is not None and alpha < 0:
            W.append(f"Negative risk-adjusted alpha of {alpha:.2%} — underperforms on a CAPM basis after adjusting for beta.")
        if high_corr:
            for t, v in sorted(high_corr.items(), key=lambda x: x[1], reverse=True)[:2]:
                W.append(f"High correlation between {t} and benchmark ({v:.2f}) reduces diversification benefit.")
        if asset_vol is not None and bench_vol is not None and asset_vol > bench_vol + 0.02:
            W.append(f"Portfolio volatility ({asset_vol:.2%}) exceeds benchmark volatility ({bench_vol:.2%}) by {(asset_vol-bench_vol):.2%}.")
        # Guarantee ≥ 3
        if len(W) < 3 and max_wt > 0.35:
            W.append(f"Concentration risk: largest position at {max_wt:.1%} weight.")
        if len(W) < 3 and var_95 is not None:
            W.append(f"Daily Value-at-Risk (95%) is {var_95:.2%} — a 1-in-20 day loss could reach this level.")
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

        # ════════════════════════════════════════════════════════════════
        # OPPORTUNITIES (blue)
        # ════════════════════════════════════════════════════════════════
        O = []
        if has_ai:
            ai_names = [t for t in tickers if t.upper() in {"NVDA", "QQQ"}]
            O.append(f"AI and semiconductor sector exposure through {' and '.join(ai_names)} positions — growth potential in AI infrastructure.")
        if tech_alloc > 0.50:
            O.append(f"Technology sector concentration ({tech_alloc:.0%}) offers upside in continued digital transformation.")
        if var_95 is not None and var_95 < -0.02:
            O.append("Tail risk management opportunity — consider protective puts or collar strategies for hedging.")
        O.append("Walk-forward rebalancing on a quarterly cadence could capture regime changes and improve out-of-sample alpha.")
        if len(sectors) < 4:
            O.append("Adding uncorrelated sectors (Healthcare, Utilities, REITs) would improve the efficient frontier.")
        if sharpe is not None and sharpe > 0 and beta is not None and beta > 1:
            O.append(f"A lower-beta variant could achieve similar risk-adjusted returns with reduced drawdown exposure.")
        # Guarantee ≥ 3
        _o_fallbacks = [
            "Options overlays (e.g., covered calls) could enhance yield and reduce effective portfolio volatility.",
            "Factor tilts (value, momentum, quality) could improve risk-adjusted returns.",
            "Systematic rebalancing triggers based on drift thresholds would enforce discipline.",
        ]
        for _fb in _o_fallbacks:
            if len(O) >= 3:
                break
            O.append(_fb)

        # ════════════════════════════════════════════════════════════════
        # THREATS (red)
        # ════════════════════════════════════════════════════════════════
        T = []
        if beta is not None and beta > 1.2:
            T.append(f"Amplified drawdown in market downturns due to high beta ({beta:.2f}x) — a 20% correction implies ~{beta*20:.0f}% portfolio loss.")
        if asset_vol is not None and bench_vol is not None and asset_vol > bench_vol + 0.05:
            T.append(f"Portfolio volatility ({asset_vol:.2%}) significantly exceeds benchmark ({bench_vol:.2%}) — elevated risk of underperformance in choppy markets.")
        if max_wt > 0.35:
            T.append(f"Concentration risk — single asset at {max_wt:.1%} creates idiosyncratic event vulnerability.")
        if dom_sector[1] > 0.60:
            T.append(f"Sector concentration ({dom_sector[0]} at {dom_sector[1]:.0%}) creates vulnerability to regulation or macro headwinds.")
        if high_corr:
            T.append(f"High benchmark correlation ({', '.join(f'{t} ρ={v:.2f}' for t,v in list(high_corr.items())[:3])}) offers limited diversification in a crash.")
        T.append("Historical optimization may not predict future regimes — structural market shifts could invalidate assumptions.")
        # Guarantee ≥ 3
        _t_fallbacks = [
            "Rising interest rates could compress equity valuations, especially for growth-oriented holdings.",
            "Geopolitical shocks or trade policy changes could disrupt sector performance.",
            "Liquidity contraction in risk-off environments may widen bid-ask spreads on smaller positions.",
        ]
        for _fb in _t_fallbacks:
            if len(T) >= 3:
                break
            T.append(_fb)

        # ════════════════════════════════════════════════════════════════
        # Render the four styled cards in a 2×2 grid
        # ════════════════════════════════════════════════════════════════
        card_data = [
            ("Strengths",       S, "#4ade80", "#0a1f0f"),
            ("Weaknesses",     W, "#fbbf24", "#1f1505"),
            ("Opportunities",  O, "#60a5fa", "#040f1f"),
            ("Threats",        T, "#f87171", "#1f0507"),
        ]

        row1 = st.columns(2)
        row2 = st.columns(2)
        grid = [row1[0], row1[1], row2[0], row2[1]]

        for i, (heading, bullets, accent, bg) in enumerate(card_data):
            with grid[i]:
                bullet_html = ""
                for b in bullets:
                    r, g, b_rgb = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
                    bullet_html += (
                        f'<div style="padding:10px 14px;border-radius:9px;margin-bottom:10px;'
                        f'background:rgba({r},{g},{b_rgb},0.07);'
                        f'border-left:3px solid {accent};color:#e2e8f0;'
                        f'font-size:0.88rem;line-height:1.55;">'
                        f'{b}</div>'
                    )
                card_html = (
                    f'<div style="border-radius:16px;padding:22px 24px;min-height:280px;'
                    f'background:{bg};border:1px solid {accent}30;margin:0 0 12px 0;'
                    f'box-shadow:0 6px 30px rgba(0,0,0,0.35);">'
                    f'<div style="font-size:1.15rem;font-weight:800;color:{accent};'
                    f'margin-bottom:14px;letter-spacing:0.5px;">{heading}</div>'
                    f'{bullet_html}</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        per_asset_swot = swot_data.get("per_asset_swot", {})
        if per_asset_swot:
            st.markdown("<br>#### Individual Asset Analysis", unsafe_allow_html=True)
            asset_cards_html = "<div style='display:flex; flex-wrap:wrap; gap:16px;'>"
            for ticker, data in per_asset_swot.items():
                s_b, w_b, o_b, t_b = [], [], [], []
                ret = data.get('cagr_1y')
                if ret is not None: s_b.append(f"1Y Return: {ret:.2%}")
                beta_a = data.get('beta')
                if beta_a is not None: s_b.append(f"Beta vs Bench: {beta_a:.2f}")
                corr_p = data.get('corr_port')
                if corr_p is not None: s_b.append(f"Port Corr: {corr_p:.2f}")
                
                vol = data.get('volatility')
                if vol is not None: w_b.append(f"Volatility: {vol:.2%}")
                mdd = data.get('max_drawdown')
                if mdd is not None: w_b.append(f"Max DD: {mdd:.2%}")
                corr_b = data.get('corr_bench')
                if corr_b is not None and corr_b > 0.9: w_b.append(f"High Bench Corr: {corr_b:.2f}")
                
                sec = data.get('sector')
                if sec and sec != "Other": o_b.append(f"Sector: {sec}")
                if ret is not None and ret > 0.2: o_b.append("Strong Momentum (>20%)")
                
                wt = data.get('weight', 0)
                if wt > 0.3: t_b.append(f"Concentration Risk ({wt:.1%})")
                if beta_a is not None and beta_a > 1.3: t_b.append(f"High Beta Risk: {beta_a:.2f}")

                if not s_b: s_b.append("Core Holding")
                if not w_b: w_b.append("Standard Risk Profile")
                if not o_b: o_b.append("Long-term Growth")
                if not t_b: t_b.append("Market Correlation")

                def _render_mini_list(color, items):
                    return "".join(f"<div style='font-size:0.75rem; margin-bottom:4px; border-left: 2px solid {color}; padding-left: 6px; color:#e2e8f0;'>{item}</div>" for item in items[:2])

                card = f"""
                <div style="flex: 1 1 200px; border-radius:12px; padding:16px; background:#111827; border:1px solid #1f2937; box-shadow:0 4px 15px rgba(0,0,0,0.2);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span style="font-weight:bold; font-size:1rem; color:#f8fafc;">{ticker}</span>
                        <span style="font-size:0.75rem; padding:2px 8px; border-radius:12px; background:#374151; color:#9ca3af;">{wt:.1%} Wt</span>
                    </div>
                    <div style="margin-bottom:8px;">
                        <div style="font-size:0.75rem; font-weight:bold; color:#4ade80; margin-bottom:4px;">Strengths</div>
                        {_render_mini_list('#4ade80', s_b)}
                    </div>
                    <div style="margin-bottom:8px;">
                        <div style="font-size:0.75rem; font-weight:bold; color:#fbbf24; margin-bottom:4px;">Weaknesses</div>
                        {_render_mini_list('#fbbf24', w_b)}
                    </div>
                    <div style="margin-bottom:8px;">
                        <div style="font-size:0.75rem; font-weight:bold; color:#60a5fa; margin-bottom:4px;">Opportunities</div>
                        {_render_mini_list('#60a5fa', o_b)}
                    </div>
                    <div>
                        <div style="font-size:0.75rem; font-weight:bold; color:#f87171; margin-bottom:4px;">Threats</div>
                        {_render_mini_list('#f87171', t_b)}
                    </div>
                </div>"""
                asset_cards_html += card
            
            asset_cards_html += "</div>"
            st.markdown(asset_cards_html, unsafe_allow_html=True)

    except Exception as swot_err:
        st.error(f"SWOT analysis rendering failed: {swot_err}")
        import traceback
        st.code(traceback.format_exc())


def _render_deep_report_sections(report_context):
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
        div[data-testid="element-container"] { margin-bottom: 0; }
        .stPlotlyChart { margin-bottom: 0; padding-bottom: 0; }
        iframe { margin-bottom: 0; }
        h2, h3, h4 { margin-top: 0.75rem; margin-bottom: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Bug 3: Show benchmark exclusion warning at the top if applicable
    if report_context.get("benchmark_excluded"):
        st.warning("⚠️ SPY excluded from portfolio optimization — it is used as benchmark.")

    section_tabs = st.tabs([section["title"] for section in report_context["sections"]])
    for tab, section in zip(section_tabs, report_context["sections"]):
        with tab:
            st.caption(section["description"])

            if section["sidebar"]:
                sidebar_cols = st.columns(len(section["sidebar"]))
                for col, block in zip(sidebar_cols, section["sidebar"]):
                    with col:
                        st.markdown(f"#### {block['title']}")
                        if block["type"] == "metrics":
                            metrics_df = pd.DataFrame(
                                list(block["data"].items()),
                                columns=["Metric", "Value"],
                            )
                            st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)
                            # Bug 3: Add clarifying caption for OOS Realized Return
                            if any("OOS Realized" in str(k) for k in block["data"].keys()):
                                st.caption("ℹ️ Realized over the out-of-sample validation period")
                        elif block["type"] == "table_html":
                            try:
                                table_df = pd.read_html(StringIO(block["data"]))[0]
                                table_df = _flatten_table_df(table_df)
                                st.dataframe(table_df, use_container_width=True)
                            except Exception:
                                st.markdown(block["data"], unsafe_allow_html=True)

            main_blocks = section["main_content"]
            i = 0
            while i < len(main_blocks):
                block = main_blocks[i]
                
                # Side-by-side layout for Allocation Chart and Table
                if block["title"] == "Portfolio Allocation Bar Chart" and i + 1 < len(main_blocks) and main_blocks[i+1]["title"] == "Portfolio Allocation Table":
                    t_block = main_blocks[i+1]
                    col1, col2 = st.columns([2, 1], gap="small")
                    
                    with col1:
                        st.markdown(f"#### {block['title']}")
                        fig = block["data"]
                        if fig is not None:
                            fig = _style_report_figure(fig, block["title"])
                            if _figure_has_data(fig):
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"{block['title']} is not available.")
                        else:
                            st.info(f"{block['title']} is not available.")
                            
                    with col2:
                        st.markdown(f"#### {t_block['title']}")
                        try:
                            table_df = pd.read_html(StringIO(t_block["data"]))[0]
                            table_df = _flatten_table_df(table_df)
                            st.dataframe(table_df, use_container_width=True, height=400)
                        except Exception:
                            st.markdown(t_block["data"], unsafe_allow_html=True)
                            
                    i += 2
                    continue

                st.markdown(f"#### {block['title']}")
                if block["type"] == "plot":
                    fig = block["data"]
                    if fig is None:
                        st.info(f"{block['title']} is not available for the current data selection.")
                        i += 1
                        continue

                    # Bug 5: Fix donut/pie chart labels for Composition charts
                    is_pie_chart = any(getattr(trace, "type", None) == "pie" for trace in fig.data)
                    if is_pie_chart:
                        fig.update_traces(
                            textposition="outside",
                            pull=[0.03 if v < 0.03 else 0 for trace in fig.data for v in (getattr(trace, 'values', []) or [])],
                        )
                        fig.update_layout(
                            height=max(getattr(fig.layout, 'height', None) or 400, 400),
                            uniformtext_minsize=8,
                            uniformtext_mode="hide",
                        )

                    fig = _style_report_figure(fig, block["title"])
                    if _figure_has_data(fig):
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"{block['title']} is not available for the current data selection.")
                elif block["type"] == "table_html":
                    try:
                        table_df = pd.read_html(StringIO(block["data"]))[0]
                        table_df = _flatten_table_df(table_df)
                        # Bug 4: Replace None/nan in rolling returns with "Insufficient data"
                        table_df = table_df.fillna("Insufficient data")
                        table_df = table_df.replace("None", "Insufficient data")
                        table_df = table_df.replace("nan", "Insufficient data")
                        st.dataframe(table_df, use_container_width=True)
                    except Exception:
                        st.markdown(block["data"], unsafe_allow_html=True)
                elif block["type"] == "swot_placeholder":
                    # Bug 12: Render SWOT section
                    _render_swot_section(block["data"])
                
                i += 1

def _render_allocation(result, tickers):
    method = result.get("method", "standard")
    dropped_tickers = result.get("dropped_tickers", [])
    if dropped_tickers:
        st.caption(f"Ignored for allocation due to missing history: {', '.join(dropped_tickers)}")

    if method == "bl":
        # --- LSTM Views Table with animated spinners ---
        st.subheader("LSTM Model Views")
        st.caption(
            "Confidence is derived from directional accuracy when an evolved LSTM is available. "
            "Otherwise the app uses a clearly labeled historical-mean fallback."
        )

        view_details = result.get("view_details", [])

        # Check for any queued/optimizing tickers
        has_optimizing = any(
            v.get("model_source") == "queued"
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
                <th>Prediction Method</th>
                <th>Model Status</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
        """

        for v in view_details:
            ret = v.get("expected_return", 0)
            ret_class = "positive" if ret >= 0 else "negative"
            ret_str = f"{ret:+.2f}%"

            source = v.get("model_source", "historical")
            prediction_label = v.get("prediction_label", "Historical mean fallback")
            if source == "evolved":
                model_cell = '<span class="evolved-badge">Evolved LSTM</span>'
            elif source == "fallback":
                model_cell = '<span class="optimizing-badge">Fallback from cached model</span>'
            elif source == "queued":
                model_cell = (
                    '<span class="optimizing-badge">'
                    '<span class="spinner"></span>Optimizing...'
                    '</span>'
                )
            else:
                model_cell = '<span class="optimizing-badge">Historical fallback</span>'

            accuracy = v.get("accuracy")
            accuracy_str = f"{accuracy}%" if accuracy else "—"

            table_html += f"""
            <tr>
                <td><strong>{v['symbol']}</strong></td>
                <td>${v['current_price']}</td>
                <td>${v['predicted_price']}</td>
                <td class="{ret_class}">{ret_str}</td>
                <td>{v['confidence']}%</td>
                <td>{prediction_label}</td>
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
            if v.get("model_source") == "queued"
        ]
        if queued:
            st.caption(
                f"Optimizing in background: {', '.join(queued)}. "
                f"Page will refresh automatically when ready."
            )

        missing_views = result.get("missing_view_tickers", [])
        if missing_views:
            st.caption(
                "No explicit BL view was generated for: "
                f"{', '.join(missing_views)}. Those assets still remain in the covariance universe."
            )

        st.divider()
        st.subheader("Recommended Allocation")
        _render_allocation_charts(result.get("allocation", {}))
        st.caption("Method: AI-Driven Black-Litterman")

    else:
        # Standard fallback
        fallback_reason = result.get("bl_unavailable_reason")
        if fallback_reason:
            st.info(f"Using risk-based optimization because Black-Litterman was unavailable: {fallback_reason}")
        else:
            st.info("Using risk-based optimization.")
        st.subheader("Recommended Allocation")
        _render_allocation_charts(result.get("allocation", {}))
        blend_meta = result.get("blend_meta", {})
        if blend_meta:
            st.caption(
                "Method: Risk-Based "
                f"({blend_meta.get('from', 'HRP')} -> {blend_meta.get('to', 'Max Sharpe')}, "
                f"alpha={blend_meta.get('alpha', 0):.2f})"
            )
        else:
            st.caption("Method: Risk-Based (inverse vol / min vol / max Sharpe)")

    # AI Reasoning
    reasoning = result.get("reasoning", "")
    if reasoning:
        st.info(f"**AI Analysis:**\n\n{reasoning}")

def _run_allocation(tickers, risk_tolerance):
    """
    Automatic allocator:
    1. Try AI-Driven Black-Litterman.
    2. Fall back to the bounded risk-based optimizer when BL is unavailable.
    """
    with st.spinner("Generating allocation..."):
        result, error = _fetch_auto_allocation(tickers, risk_tolerance, target_date=target_date)
        if result:
            st.session_state["allocation_result"] = result
            _render_allocation(result, tickers)
        else:
            st.error(f"Optimization failed: {error}")

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
            "AAPL, MSFT, QQQ, NVDA",
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
            st.dataframe(w_df.set_index("Ticker"), width='stretch')

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
            benchmark_info = bl_res.get("benchmark", {})
            benchmark_label = benchmark_info.get("label", benchmark_info.get("ticker", "Benchmark"))
            sample_window = bl_res.get("benchmark_window", {})
            default_realized_window = bl_res.get("realized_window", {})
            st.caption(
                f"Delta (risk aversion): {bl_res['delta']} | "
                f"Tau: {bl_res['tau']} | "
                f"Risk-free rate: {bl_res['risk_free_rate']}"
            )
            if sample_window:
                st.caption(
                    f"Benchmark sample for beta and equilibrium returns: {benchmark_label} "
                    f"from {sample_window.get('start')} to {sample_window.get('end')} "
                    f"({sample_window.get('trading_days', 0)} trading days)."
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
            st.plotly_chart(fig_weights, width='stretch')

            # --- Time period performance summary ---
            st.subheader("Time Period Performance")
            prices_df = _history_payload_to_frame(bl_res.get("price_history"))
            benchmark_history_df = _history_payload_to_frame(bl_res.get("benchmark_history"))
            if not prices_df.empty:
                min_history_date = prices_df.index.min().date()
                max_history_date = prices_df.index.max().date()
                default_from = (
                    pd.to_datetime(default_realized_window["start"]).date()
                    if default_realized_window.get("start")
                    else max(min_history_date, max_history_date - timedelta(days=45))
                )
                default_to = (
                    pd.to_datetime(default_realized_window["end"]).date()
                    if default_realized_window.get("end")
                    else max_history_date
                )

                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    perf_from = st.date_input(
                        "From Date",
                        value=default_from,
                        min_value=min_history_date,
                        max_value=max_history_date,
                        key="bl_perf_from",
                    )
                with perf_col2:
                    perf_to = st.date_input(
                        "To Date",
                        value=default_to,
                        min_value=min_history_date,
                        max_value=max_history_date,
                        key="bl_perf_to",
                    )

                if perf_from >= perf_to:
                    st.warning("Choose a valid performance range where the from date is before the to date.")
                else:
                    selected_prices = prices_df.loc[str(perf_from):str(perf_to)]
                    if len(selected_prices) < 2:
                        st.warning("Selected range does not contain enough data points.")
                    else:
                        window_label = f"{perf_from} to {perf_to}"
                        benchmark_series = None
                        if not benchmark_history_df.empty:
                            selected_benchmark = benchmark_history_df.loc[str(perf_from):str(perf_to)]
                            if len(selected_benchmark) >= 2:
                                benchmark_series = selected_benchmark.iloc[:, 0]

                        ret_df = _build_bl_return_frame(bl_res, tks)
                        asset_window_returns = _asset_window_returns(selected_prices[tks])
                        benchmark_return_pct = _window_return_percent(benchmark_series)
                        ret_df["realized_return_pct"] = ret_df["ticker"].map(asset_window_returns)
                        ret_df["benchmark_return_pct"] = benchmark_return_pct
                        ret_df["excess_vs_benchmark_pct"] = ret_df["realized_return_pct"].apply(
                            lambda val: (
                                round(float(val - benchmark_return_pct), 2)
                                if val is not None and benchmark_return_pct is not None
                                else None
                            )
                        )
                        ret_df = ret_df.rename(columns={
                            "ticker": "Ticker",
                            "beta_vs_benchmark": f"Beta vs {benchmark_label}",
                            "equilibrium_return_pct": "Equilibrium Return %",
                            "posterior_return_pct": "Posterior Return %",
                            "market_weight": "Market Weight",
                            "bl_weight": "BL Weight",
                            "weight_tilt": "Weight Tilt",
                            "realized_return_pct": f"Realized Return % ({window_label})",
                            "benchmark_return_pct": f"{benchmark_label} Return % ({window_label})",
                            "excess_vs_benchmark_pct": f"Excess vs {benchmark_label} % ({window_label})",
                        })

                        for column in ["Market Weight", "BL Weight", "Weight Tilt"]:
                            if column in ret_df.columns:
                                ret_df[column] = ret_df[column].map(_format_weight_fraction)

                        def _color_signed(val):
                            try:
                                normalized = str(val).replace("%", "").strip()
                                if not normalized:
                                    return ""
                                v = float(normalized)
                                if v > 0:
                                    return "color: #4CAF50"
                                if v < 0:
                                    return "color: #f44336"
                            except (ValueError, TypeError):
                                return ""
                            return ""

                        st.subheader("Expected vs Realized Comparison")
                        styled_ret_df = ret_df.set_index("Ticker").style
                        signed_subset = [
                            column for column in [
                                "Weight Tilt",
                                f"Excess vs {benchmark_label} % ({window_label})",
                            ]
                            if column in ret_df.columns
                        ]
                        if signed_subset:
                            styled_ret_df = styled_ret_df.applymap(
                                _color_signed,
                                subset=signed_subset,
                            )
                        st.dataframe(styled_ret_df, width='stretch')
                        st.caption(
                            f"Beta is estimated from the benchmark sample above versus {benchmark_label}. "
                            f"Realized return columns use the selected window ({window_label}). "
                            "Omega, shown below, is the view-uncertainty measure."
                        )

                        equal_weight = {t: 1.0 / len(tks) for t in tks}
                        bl_growth = _portfolio_growth(selected_prices, bl_w)
                        eq_growth = _portfolio_growth(selected_prices, equal_weight)
                        mcap_growth = _portfolio_growth(selected_prices, eq_w)
                        bl_metrics = _performance_metrics(bl_growth)

                        if bl_metrics:
                            mc1, mc2, mc3 = st.columns(3)
                            with mc1:
                                st.metric("Total Return", f"{bl_metrics['total_return']:.2%}")
                            with mc2:
                                st.metric("Annualized Return", f"{bl_metrics['annualized_return']:.2%}")
                            with mc3:
                                st.metric(
                                    "Max Drawdown",
                                    f"{bl_metrics['max_drawdown']:.2%}",
                                    delta_color="inverse",
                                )

                            base_capital = 10000.0
                            pnl_value = base_capital * bl_metrics["total_return"]
                            st.caption(
                                f"Actual BL profit/loss over the selected period on a "
                                f"${base_capital:,.0f} starting value: {pnl_value:+,.2f}"
                            )

                            comparison_df = pd.DataFrame({
                                "BL Optimal": bl_growth,
                                "Equal Weight": eq_growth,
                                "Market Cap Weight": mcap_growth,
                            }).dropna()
                            if benchmark_series is not None:
                                comparison_df[benchmark_label] = benchmark_series / benchmark_series.iloc[0]

                            perf_fig = go.Figure()
                            for name, color in [
                                ("BL Optimal", "#00CC96"),
                                ("Equal Weight", "#636EFA"),
                                ("Market Cap Weight", "#FFA15A"),
                                (benchmark_label, "#FECB52"),
                            ]:
                                if name not in comparison_df:
                                    continue
                                perf_fig.add_trace(go.Scatter(
                                    x=comparison_df.index,
                                    y=comparison_df[name],
                                    mode="lines",
                                    name=name,
                                    line=dict(width=3 if name == "BL Optimal" else 2, color=color),
                                ))
                            perf_fig.update_layout(
                                title="Cumulative Returns Over Selected Period",
                                xaxis_title="Date",
                                yaxis_title="Growth of $1",
                                hovermode="x unified",
                                margin=dict(t=50, b=30),
                            )
                            st.plotly_chart(perf_fig, width='stretch')

            # --- Allocation impact from BL views ---
            st.subheader("Allocation Tilt From Views")
            tilt_vals = [bl_w.get(t, 0) - eq_w.get(t, 0) for t in tks]
            tilt_fig = go.Figure()
            tilt_fig.add_trace(go.Bar(
                name="BL - Market Weight",
                x=tks,
                y=tilt_vals,
                marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in tilt_vals],
                text=[f"{v:+.2%}" for v in tilt_vals],
                textposition="outside",
            ))
            tilt_fig.update_layout(
                yaxis_title="Weight Change",
                xaxis_title="Ticker",
                margin=dict(t=30, b=30),
                showlegend=False,
            )
            st.plotly_chart(tilt_fig, width='stretch')
            st.caption("Positive bars are overweight versus the market portfolio; negative bars are underweight.")

            # --- View signal chart ---
            st.subheader("View Signal Decomposition")
            view_decomp = bl_res.get("view_decomposition", [])
            if view_decomp:
                decomp_fig = go.Figure()
                colors = ["#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
                for idx, vd in enumerate(view_decomp):
                    lam = vd["lambda"]
                    vw = vd["weights"]
                    direction = -1 if lam < 0 else 1
                    decomp_fig.add_trace(go.Bar(
                        name=f"View {idx + 1} (λ={lam:.4f})",
                        x=tks,
                        y=[vw.get(t, 0) * direction for t in tks],
                        marker_color=colors[idx % len(colors)],
                        hovertemplate=(
                            f"View {idx + 1}<br>"
                            "Ticker=%{x}<br>"
                            "Directional exposure=%{y:.4f}<br>"
                            f"Lambda={lam:.4f}<extra></extra>"
                        ),
                    ))

                decomp_fig.update_layout(
                    barmode="group",
                    yaxis_title="Normalized Directional Exposure",
                    xaxis_title="Ticker",
                    margin=dict(t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(decomp_fig, width='stretch')
                st.caption(
                    "This shows the normalized long/short direction of each view. "
                    "Lambda is a view-strength coefficient, not a portfolio weight."
                )

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
                st.dataframe(iv_df.set_index("View"), width='stretch')

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
                st.plotly_chart(fig_omega, width='stretch')
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
        st.write(
            "Generate a comprehensive report with backtesting, efficient frontier, "
            "walk-forward validation, and Monte Carlo analysis."
        )

        REPORT_VERSION = "v3"
        if st.session_state.get("deep_report_version") != REPORT_VERSION:
            st.session_state.pop("deep_report_context", None)
            st.session_state["deep_report_version"] = REPORT_VERSION
        
        with st.expander("Report Configuration", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, QQQ, NVDA")
                benchmark = st.text_input("Benchmark Ticker", "SPY")
            with col_in2:
                default_start = datetime.now() - timedelta(days=365*2)
                start_date = st.date_input("Training Start Date", default_start)
                end_date = st.date_input("Training End Date", datetime.now() - timedelta(days=90))
                report_risk_tolerance = st.slider(
                    "Risk Tolerance for Report Allocation",
                    0.0, 1.0, 0.5, 0.05,
                    key="deep_report_risk",
                )

        generate_report = st.button("Generate Deep Report", type="primary")

        if generate_report:
            st.session_state["deep_report_context"] = None
            r_tickers = [t.strip() for t in repo_tickers.split(",") if t.strip()]
            if not r_tickers:
                st.warning("Enter tickers.")
            else:
                report_started_at = time.time()
                report_progress_placeholder = st.empty()
                _render_deep_report_progress(
                    report_progress_placeholder,
                    0.03,
                    "Preparing request",
                    "Validating tickers and report dates.",
                    report_started_at,
                    eta_seconds=60,
                )

                _render_deep_report_progress(
                    report_progress_placeholder,
                    0.10,
                    "Validating benchmark",
                    f"Checking benchmark history for {benchmark.strip().upper() or 'the selected symbol'}.",
                    report_started_at,
                    eta_seconds=55,
                )
                validated_benchmark, benchmark_error = _validate_benchmark_ticker(
                    benchmark,
                    start_date,
                    end_date,
                )
                if benchmark_error:
                    report_progress_placeholder.empty()
                    st.error(benchmark_error)
                else:
                    _render_deep_report_progress(
                        report_progress_placeholder,
                        0.16,
                        "Fetching allocation",
                        "Requesting the starting portfolio weights from the backend.",
                        report_started_at,
                        eta_seconds=50,
                    )
                    allocation_result, alloc_error = _fetch_auto_allocation(r_tickers, report_risk_tolerance, target_date=target_date)
                    if alloc_error or not allocation_result:
                        equal_weights = {t: 1.0 / len(r_tickers) for t in r_tickers}
                        allocation_result = {
                            "allocation": equal_weights,
                            "source": "risk_based",
                            "blend_meta": {},
                            "fallback_reason": alloc_error or "Allocation service unavailable",
                            "requested_tickers": r_tickers,
                            "valid_tickers": r_tickers,
                            "dropped_tickers": [],
                            "method": "standard",
                            "bl_unavailable_reason": alloc_error,
                        }
                        st.info(
                            f"Using equal-weight allocation because automated allocation was unavailable: "
                            f"{allocation_result['fallback_reason']}."
                        )
                    if True:
                        report_file = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..",
                            "portfolio_report.html"
                        )
                        report_file = os.path.normpath(report_file)
                        _render_deep_report_progress(
                            report_progress_placeholder,
                            0.22,
                            "Collecting asset metadata",
                            "Fetching sector details used in the report visuals.",
                            report_started_at,
                            eta_seconds=47,
                        )
                        sector_map = _fetch_sector_map(r_tickers)
                        try:
                            report_context = _build_deep_report_context(
                                r_tickers,
                                allocation_result,
                                validated_benchmark,
                                start_date,
                                end_date,
                                report_file,
                                sector_map,
                                report_risk_tolerance,
                                progress_callback=lambda progress_value, stage_label, detail, eta_seconds=None: _render_deep_report_progress(
                                    report_progress_placeholder,
                                    progress_value,
                                    stage_label,
                                    detail,
                                    report_started_at,
                                    eta_seconds=eta_seconds,
                                ),
                            )
                            st.session_state["deep_report_context"] = report_context
                            st.success(
                                f"Report generated successfully in {_format_duration(time.time() - report_started_at)}."
                            )
                        except Exception as e:
                            import traceback
                            st.session_state["deep_report_context"] = None
                            st.error(f"Report generation failed: {e}")
                            st.code(traceback.format_exc())

        report_context = st.session_state.get("deep_report_context")
        if report_context:
            st.divider()
            st.subheader("Deep Report Dashboard")
            allocation_result = report_context.get("allocation_result", {})
            report_universe = report_context.get("report_universe", {})
            allocation_source = allocation_result.get("source", "risk_based")
            allocation_source_label = (
                "AI-Driven Black-Litterman"
                if allocation_source == "bl"
                else "Risk-Based Fallback"
            )
            st.caption(
                f"Risk-free rate used in the report: {report_context['risk_free_rate']:.2%} | "
                f"Allocation source: {allocation_source_label}"
            )
            if report_universe.get("dropped_tickers"):
                st.caption(
                    "Unavailable for the shared report universe: "
                    f"{', '.join(report_universe['dropped_tickers'])}"
                )
            st.download_button(
                label="Download HTML Report",
                data=report_context["html_content"],
                file_name="My_Quant_Report.html",
                mime="text/html",
            )
            _render_deep_report_sections(report_context)

            with st.expander("Legacy HTML Preview", expanded=False):
                components.html(report_context["html_content"], height=800, scrolling=True)

