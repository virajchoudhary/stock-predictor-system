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
from io import StringIO
from datetime import datetime, timedelta
import streamlit.components.v1 as components

_FRONTEND_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if _FRONTEND_ROOT not in sys.path:
    sys.path.insert(0, _FRONTEND_ROOT)

from reporting_utils import resolve_report_universe

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
        font=dict(size=16, color="#FAFAFA"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(t=60, b=30, l=30, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
    )
    return fig


def _style_report_figure(fig, fallback_title, xaxis_title=None, yaxis_title=None, height=430):
    if fig is None:
        return None

    title_obj = fig.layout.title.text if getattr(fig.layout, "title", None) else None
    fig.update_layout(
        title=title_obj or fallback_title,
        height=height,
        margin=dict(t=60, b=30, l=30, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    has_cartesian_axes = any(getattr(trace, "type", None) not in {"pie"} for trace in fig.data)
    if has_cartesian_axes:
        if xaxis_title:
            fig.update_xaxes(title_text=xaxis_title)
        elif not fig.layout.xaxis.title.text:
            fig.update_xaxes(title_text="Date")

        if yaxis_title:
            fig.update_yaxes(title_text=yaxis_title)

    return fig


def _prepare_report_figure(fig, title, xaxis_title=None, yaxis_title=None, height=430):
    if fig is None or not _figure_has_data(fig):
        return _empty_report_figure(title, f"{title} is not available for the current data selection.")
    return _style_report_figure(
        fig,
        fallback_title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
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


def _request_bl_allocation(tickers, risk_tolerance):
    try:
        response = requests.post(
            f"{API_URL}/bl-optimize/",
            json={"tickers": tickers, "risk_tolerance": risk_tolerance},
            timeout=45,
        )
        if response.status_code != 200:
            return None, _response_error_message(response)
        return response.json(), None
    except requests.Timeout:
        return None, "The Black-Litterman allocation service timed out."
    except Exception as exc:
        return None, str(exc)


def _request_risk_allocation(tickers, risk_tolerance):
    try:
        response = requests.post(
            f"{API_URL}/optimize/",
            json={"tickers": tickers, "risk_tolerance": risk_tolerance},
            timeout=20,
        )
        if response.status_code != 200:
            return None, _response_error_message(response)
        return response.json(), None
    except requests.Timeout:
        return None, "The risk-based allocation service timed out."
    except Exception as exc:
        return None, str(exc)


def _fetch_auto_allocation(tickers, risk_tolerance):
    risk_result, risk_error = _request_risk_allocation(tickers, risk_tolerance)
    if risk_result and risk_result.get("allocation"):
        risk_result["method"] = "standard"
        return risk_result, None

    return None, risk_error or "Allocation services were unavailable."


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
    risk_free_rate = qr.opt_core.get_risk_free_rate()
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
    pr_rolling_html = calculate_rolling_returns(portfolio_eval).to_html(classes="metrics-table")
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
    if report_universe["dropped_tickers"]:
        allocation_metrics["Unavailable Tickers"] = ", ".join(report_universe["dropped_tickers"])
    if allocation_fallback_reason:
        allocation_metrics["Fallback Reason"] = allocation_fallback_reason

    allocation_section = {
        "title": "Portfolio Construction",
        "description": (
            f"{allocation_source_label} allocation for the requested report universe "
            f"using benchmark {benchmark_ticker}."
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
            block["data"] = _prepare_report_figure(
                block["data"],
                title=block["title"],
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                height=470 if "Composition" in block["title"] else 430,
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
    }


def _render_deep_report_sections(report_context):
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
                            st.dataframe(metrics_df.set_index("Metric"), width='stretch')
                        elif block["type"] == "table_html":
                            try:
                                table_df = pd.read_html(StringIO(block["data"]))[0]
                                st.dataframe(table_df, width='stretch')
                            except Exception:
                                st.markdown(block["data"], unsafe_allow_html=True)

            for block in section["main_content"]:
                st.markdown(f"#### {block['title']}")
                if block["type"] == "plot":
                    fig = _style_report_figure(block["data"], block["title"])
                    if _figure_has_data(fig):
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info(f"{block['title']} is not available for the current data selection.")
                elif block["type"] == "table_html":
                    try:
                        table_df = pd.read_html(StringIO(block["data"]))[0]
                        st.dataframe(table_df, width='stretch')
                    except Exception:
                        st.markdown(block["data"], unsafe_allow_html=True)

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
        result, error = _fetch_auto_allocation(tickers, risk_tolerance)
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
        "Generates an optimal allocation using **Riskfolio-Lib** "
        "optimization. Adjust the slider to scale smoothly between "
        "Conservative (Minimum Volatility), Balanced (Max Sharpe), and Aggressive "
        "(Max Utility) portfolios."
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
        
        with st.expander("Report Configuration", expanded=True):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                repo_tickers = st.text_input("Portfolio Tickers", "AAPL, MSFT, SPY, QQQ")
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
                    allocation_result, alloc_error = _fetch_auto_allocation(r_tickers, report_risk_tolerance)
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

