"""
Embeddable "Model Validation (Proof of Accuracy)" expander.

Mount at the bottom of any page by calling:
    render_model_validation_expander(page_prefix="pf", default_symbol=symbol)

All Streamlit widget keys are uniquely namespaced per page via `page_prefix`
to strictly avoid DuplicateWidgetID crashes when the expander is reused.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _metric_card(label: str, value: str, color: str | None = None):
    color_css = f"color:{color};" if color else ""
    st.markdown(
        f"""
        <div style="padding:0.75rem 1rem;border:1px solid #2b2b2b;border-radius:8px;
                    background:#111;">
          <div style="font-size:0.75rem;color:#888;text-transform:uppercase;
                      letter-spacing:0.08em;">{label}</div>
          <div style="font-size:1.35rem;font-weight:600;margin-top:0.25rem;{color_css}">
            {value}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_validation_expander(page_prefix: str, default_symbol: str = "AAPL"):
    """
    Render the Model Validation expander.

    Args:
        page_prefix: short string used to namespace widget keys
                     (e.g. "dash", "pf", "opt"). Must be unique per page.
        default_symbol: seed value for the symbol input.
    """
    k = lambda name: f"{page_prefix}_val_{name}"  # noqa: E731

    with st.expander("Model Validation (Proof of Accuracy)", expanded=False):
        st.caption(
            "Truncate history to a past date, retrain the Bidirectional LSTM from "
            "scratch, and blindly predict the path to today. Metrics use T+1 "
            "execution with transaction fee + slippage."
        )

        col_sym, col_date, col_force = st.columns([1.3, 1.2, 1.0])
        with col_sym:
            symbol = st.text_input(
                "Symbol",
                value=default_symbol,
                key=k("symbol"),
                placeholder="e.g. AAPL",
            )
        with col_date:
            past_date = st.date_input(
                "Past Date",
                value=date.today() - timedelta(days=30),
                max_value=date.today() - timedelta(days=7),
                key=k("date"),
            )
        with col_force:
            force_retrain = st.checkbox(
                "Force Retrain (Ignore Cache)",
                value=True,
                key=k("force"),
            )

        submit = st.button(
            "Run Model Validation",
            type="primary",
            key=k("submit"),
            use_container_width=True,
        )

        result_state_key = k("result")
        if submit:
            if not symbol:
                st.error("Enter a symbol to validate.")
                return
            with st.spinner(
                f"Retraining BiLSTM on data up to {past_date} and rolling forward..."
            ):
                try:
                    resp = requests.get(
                        f"{API_URL}/backtest-predict/{symbol.upper()}/",
                        params={
                            "past_date": past_date.strftime("%Y-%m-%d"),
                            "force_retrain": str(force_retrain).lower(),
                        },
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        st.session_state[result_state_key] = resp.json()
                    else:
                        try:
                            err = resp.json().get("error", resp.text)
                        except Exception:
                            err = resp.text
                        st.error(f"Backend error: {err}")
                        st.session_state.pop(result_state_key, None)
                except Exception as exc:
                    st.error(f"Connection error: {exc}")
                    st.session_state.pop(result_state_key, None)

        data = st.session_state.get(result_state_key)
        if not data:
            st.info("Pick a past date and click **Run Model Validation** to begin.")
            return

        if data.get("error"):
            st.error(data["error"])
            return

        # ------------- Plotly chart -------------
        hist_ctx = pd.DataFrame(data.get("historical_context", []))
        actual   = pd.DataFrame(data.get("actual_path", []))
        predicted = pd.DataFrame(data.get("predicted_path", []))

        fig = go.Figure()
        if not hist_ctx.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(hist_ctx["date"]),
                y=hist_ctx["close"],
                mode="lines",
                name="Historical Context",
                line=dict(color="#999999", width=2),
            ))
        if not actual.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(actual["date"]),
                y=actual["close"],
                mode="lines",
                name="Actual Market Path",
                line=dict(color="#2ecc71", width=2.5),
            ))
        if not predicted.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(predicted["date"]),
                y=predicted["close"],
                mode="lines",
                name="LSTM Prediction",
                line=dict(color="#e67e22", width=2.5, dash="dash"),
            ))
        _seam = pd.to_datetime(data["past_date"]).strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=_seam, x1=_seam,
            xref="x",
            y0=0, y1=1,
            yref="paper",
            line=dict(color="#666", dash="dot"),
        )
        fig.add_annotation(
            x=_seam, y=1.0, xref="x", yref="paper",
            text="Train/Test Seam", showarrow=False,
            yanchor="bottom", font=dict(color="#888", size=11),
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_dark",
            legend=dict(orientation="h", y=1.05),
            xaxis_title="Date",
            yaxis_title="Close",
        )
        st.plotly_chart(fig, use_container_width=True, key=k("chart"))

        # ------------- Metrics -------------
        m = data.get("metrics", {})
        rmse = float(m.get("rmse", 0.0))
        mae  = float(m.get("mae", 0.0))
        dir_acc = float(m.get("directional_accuracy", 0.0))
        realized = float(m.get("expected_realized_return_net", 0.0))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _metric_card("RMSE", f"{rmse:.3f}")
        with c2:
            _metric_card("MAE", f"{mae:.3f}")
        with c3:
            dir_color = "#2ecc71" if dir_acc >= 50 else "#e74c3c"
            _metric_card("Directional Accuracy", f"{dir_acc:.2f}%", color=dir_color)
        with c4:
            realized_color = "#2ecc71" if realized >= 0 else "#e74c3c"
            _metric_card(
                "Expected Realized Return (Net)",
                _fmt_pct(realized),
                color=realized_color,
            )

        st.caption(
            f"Horizon: {data.get('horizon_days', 0)} trading days | "
            f"Training rows: {data.get('training_rows', 0)} | "
            f"Fee {m.get('transaction_fee', 0)*100:.2f}% + "
            f"Slippage {m.get('slippage', 0)*100:.2f}% per leg, T+1 execution"
        )

        # ---------------- Stacked ensemble diagnostics ----------------
        ens = data.get("ensemble") or {}
        if ens:
            w = ens.get("stacking_weights", {}) or {}
            rows = [
                ("Bi-LSTM (×{0})".format(ens.get("lstm_members", 0)),
                 "—",
                 ens.get("lstm_val_mae", 0.0),
                 w.get("lstm", 0.0)),
                ("Random Forest",
                 ens.get("rf_best_depth", "—"),
                 ens.get("rf_val_mae", 0.0),
                 w.get("rf", 0.0)),
                (ens.get("gbm_type", "GBM"),
                 ens.get("gbm_best_depth", "—"),
                 ens.get("gbm_val_mae", 0.0),
                 w.get("gbm", 0.0)),
            ]
            diag_df = pd.DataFrame(
                rows,
                columns=["Model", "Best Depth", "Val MAE", "Stack Weight"],
            )
            diag_df["Val MAE"]      = diag_df["Val MAE"].map(lambda v: f"{float(v):.5f}")
            diag_df["Stack Weight"] = diag_df["Stack Weight"].map(lambda v: f"{float(v)*100:.1f}%")
            st.markdown("**Stacked Ensemble Diagnostics**")
            st.dataframe(diag_df, hide_index=True, use_container_width=True)
            st.caption(
                f"Daily drift (blended, capped): {data.get('drift_daily', 0.0)*100:.3f}% | "
                f"Historical daily vol: {data.get('hist_vol_daily', 0.0)*100:.3f}% | "
                "Weights are inverse-MAE normalised on the held-out validation slice."
            )
