import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import BlackLittermanModel, risk_models, expected_returns
from pypfopt.black_litterman import market_implied_prior_returns
from .ai_services import TrendPredictor, get_price_history
from .models import OptimizedHyperparams


DEFAULT_CONFIDENCE = 0.3
MIN_CONFIDENCE     = 0.1
MAX_CONFIDENCE     = 0.9


def get_market_caps(tickers):
    """
    Fetch market cap for each ticker via yfinance.
    Falls back to equal weighting if market cap unavailable.
    Returns dict {symbol: market_cap}.
    """
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap  = info.get("marketCap") or info.get("market_cap")
            if cap and cap > 0:
                caps[t] = float(cap)
        except Exception:
            pass

    if not caps:
        return {t: 1.0 for t in tickers}

    # fill missing with average
    avg = np.mean(list(caps.values()))
    for t in tickers:
        if t not in caps:
            caps[t] = avg

    return caps


def build_views(tickers):
    """
    For each ticker:
    - Fetch current price
    - Use evolved LSTM forecasts when available
    - Fall back to historical mean return forecasts otherwise
    - Set confidence from directional_accuracy if evolved, else DEFAULT_CONFIDENCE

    Returns:
        views: dict {symbol: expected_return_float}
        confidences: dict {symbol: confidence_float}
        view_details: list of dicts for frontend display
    """
    views       = {}
    confidences = {}
    view_details = []

    for symbol in tickers:
        try:
            # get current price
            hist = get_price_history(symbol, period="5d")
            if hist is None or hist.empty:
                continue
            close_series = TrendPredictor._get_close_series(hist)
            if close_series.empty:
                continue
            current_price = float(close_series.iloc[-1])

            # check for evolved hyperparams
            cached = OptimizedHyperparams.get_valid_cache(symbol)

            if cached:
                forecast = TrendPredictor.forecast_price(
                    symbol,
                    hidden_size    = cached.hidden_size,
                    num_layers     = cached.num_layers,
                    learning_rate  = cached.learning_rate,
                    epochs         = cached.epochs,
                    dropout        = cached.dropout,
                    seq_len        = cached.seq_len,
                    allow_lstm     = True,
                )
                # scale accuracy to confidence — clip to [MIN, MAX]
                raw_conf   = cached.directional_accuracy / 100
                confidence = float(np.clip(raw_conf, MIN_CONFIDENCE, MAX_CONFIDENCE))
                model_source = "evolved"
                accuracy     = cached.directional_accuracy
            else:
                # No evolved model — queue GA immediately in background if Redis is up.
                try:
                    from .tasks import queue_ga_for_ticker
                    queued = queue_ga_for_ticker(symbol)
                except Exception:
                    queued = False

                forecast = TrendPredictor.forecast_price(symbol, allow_lstm=False)
                confidence      = DEFAULT_CONFIDENCE
                model_source    = "queued" if queued else "historical"
                accuracy        = None

            if not forecast:
                continue

            predicted_price = float(forecast["predicted_price"])
            expected_return = float(forecast["expected_return"])
            prediction_method = forecast.get("prediction_method", "historical_mean_fallback")
            prediction_label = forecast.get("prediction_label", "Historical mean fallback")
            if prediction_method != "lstm":
                confidence = min(confidence, DEFAULT_CONFIDENCE)
                if cached:
                    model_source = "fallback"

            views[symbol]       = expected_return
            confidences[symbol] = confidence

            view_details.append({
                "symbol":           symbol,
                "current_price":    round(current_price, 2),
                "predicted_price":  round(predicted_price, 2),
                "expected_return":  round(expected_return * 100, 2),
                "confidence":       round(confidence * 100, 1),
                "model_source":     model_source,  # "evolved", "queued", or "default"
                "accuracy":         round(accuracy, 1) if accuracy else None,
                "prediction_method": prediction_method,
                "prediction_label": prediction_label,
            })

        except Exception:
            continue

    return views, confidences, view_details


def run_black_litterman(tickers):
    """
    Full BL pipeline:
    1. Fetch price history and compute covariance matrix
    2. Fetch market caps and compute market-implied prior returns
    3. Build LSTM views and confidence levels
    4. Run BlackLittermanModel
    5. Return allocation weights, views detail, and BL expected returns

    Returns dict with keys:
        allocation: {symbol: weight}
        view_details: list of view dicts
        bl_returns: {symbol: expected_return_%}
        error: str or None
    """
    requested_tickers = [str(t).upper() for t in tickers]
    try:
        # --- Price data ---
        frames = {}
        for t in requested_tickers:
            hist = get_price_history(t, period="2y")
            if hist is not None and not hist.empty:
                close_series = TrendPredictor._get_close_series(hist)
                if not close_series.empty:
                    frames[t] = close_series

        if len(frames) < 2:
            return {"error": "Need at least 2 tickers with sufficient price history."}

        prices = pd.DataFrame(frames).dropna()
        if len(prices) < 60:
            return {"error": "Not enough historical data (need 60+ days)."}

        # only keep tickers we have data for
        valid_tickers = list(prices.columns)

        # --- Covariance matrix ---
        cov_matrix = risk_models.sample_cov(prices)

        # --- Market caps and prior returns ---
        market_caps = pd.Series(get_market_caps(valid_tickers)).reindex(valid_tickers).fillna(1.0)
        market_weights = market_caps / market_caps.sum()

        delta       = 2.5  # market risk aversion — standard value
        prior       = market_implied_prior_returns(
            market_caps, delta, cov_matrix
        )

        # --- LSTM views ---
        views, confidences, view_details = build_views(valid_tickers)

        if not views:
            return {"error": "Could not generate LSTM views for any ticker."}

        # --- Omega (uncertainty) matrix ---
        # Omega diagonal = (1 - confidence) * variance of view portfolio
        # PyPortfolioOpt handles this via intervals — we build manually
        omega_diag = []
        view_symbols = list(views.keys())
        for sym in view_symbols:
            conf      = confidences.get(sym, DEFAULT_CONFIDENCE)
            # higher confidence → lower uncertainty
            variance  = float(cov_matrix.loc[sym, sym]) if sym in cov_matrix.index else 0.01
            omega_val = (1 - conf) * variance
            omega_diag.append(max(omega_val, 1e-6))  # floor to avoid singularity

        # P matrix: one row per view, identity for absolute views
        P = np.zeros((len(view_symbols), len(valid_tickers)))
        for row_idx, sym in enumerate(view_symbols):
            if sym in valid_tickers:
                P[row_idx, valid_tickers.index(sym)] = 1.0

        Q = np.array([views[sym] for sym in view_symbols], dtype=float).reshape(-1, 1)
        omega  = np.diag(omega_diag)

        # --- Black-Litterman model ---
        bl = BlackLittermanModel(
            cov_matrix,
            pi        = prior,
            P         = P,
            Q         = Q,
            omega     = omega,
            tau       = 0.05,
            risk_aversion = delta
        )

        bl_returns = bl.bl_returns()

        # --- Mean-Variance optimization on BL returns ---
        from pypfopt import EfficientFrontier
        ef = EfficientFrontier(bl_returns, cov_matrix)
        ef.max_sharpe(risk_free_rate=0.05)
        weights = ef.clean_weights()

        allocation = {
            ticker: round(float(weights.get(ticker, 0.0)), 4)
            for ticker in valid_tickers
        }

        bl_returns_pct = {
            ticker: round(float(bl_returns.get(ticker, 0.0)) * 100, 2)
            for ticker in valid_tickers
        }
        missing_view_tickers = [ticker for ticker in valid_tickers if ticker not in views]

        return {
            "allocation":   allocation,
            "view_details": view_details,
            "bl_returns":   bl_returns_pct,
            "requested_tickers": requested_tickers,
            "valid_tickers": valid_tickers,
            "dropped_tickers": [ticker for ticker in requested_tickers if ticker not in valid_tickers],
            "view_tickers": view_symbols,
            "missing_view_tickers": missing_view_tickers,
            "source": "bl",
            "fallback_reason": None,
            "error":        None
        }

    except Exception as e:
        return {"error": str(e)}
