"""
Black-Litterman model implemented from scratch with NumPy.

Implements:
- Ledoit-Wolf covariance estimation
- Delta (risk aversion) computed from market data
- Implied equilibrium returns (Pi)
- P/Q view matrices with absolute and relative views
- Idzorek's confidence-based Omega calibration
- BL posterior expected returns and covariance
- View portfolio decomposition
- Reverse BL (implied views from weights)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
from sklearn.covariance import LedoitWolf

try:
    _YF_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "yfinance"
    _YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_YF_CACHE_DIR))
except Exception:
    pass

BENCHMARK_LABELS = {
    "^GSPC": "S&P 500",
    "^NSEI": "Nifty 50",
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_prices(tickers, years=2):
    """Fetch daily close prices for *tickers* over *years* years via yfinance."""
    frames = {}
    for t in tickers:
        try:
            hist = yf.download(t, period=f"{years}y", progress=False)
            if hist is not None and len(hist) > 60:
                close = hist["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                frames[t] = close
        except Exception:
            pass
    if not frames:
        return None
    prices = pd.DataFrame(frames).dropna()
    return prices if len(prices) > 60 else None


def fetch_market_caps(tickers):
    """Return dict {ticker: market_cap}. Falls back to equal weight."""
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap") or info.get("market_cap")
            if cap and cap > 0:
                caps[t] = float(cap)
        except Exception:
            pass
    if not caps:
        return {t: 1.0 for t in tickers}
    avg = np.mean(list(caps.values()))
    for t in tickers:
        if t not in caps:
            caps[t] = avg
    return caps


def select_market_benchmark(tickers):
    indian_count = sum(
        1 for t in tickers
        if t.upper().endswith(".NS") or t.upper().endswith(".BO")
    )
    if indian_count >= max(1, len(tickers) / 2):
        return "^NSEI"
    return "^GSPC"


def fetch_benchmark_prices(benchmark_ticker, years=2):
    try:
        hist = yf.download(benchmark_ticker, period=f"{years}y", progress=False)
        if hist is None or hist.empty:
            return None
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        benchmark_prices = close.dropna()
        return benchmark_prices if len(benchmark_prices) > 60 else None
    except Exception:
        return None


def compute_asset_betas(prices, benchmark_prices):
    asset_returns = prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    benchmark_returns.name = "benchmark"

    aligned = asset_returns.join(benchmark_returns, how="inner").dropna()
    if aligned.empty:
        return {t: 0.0 for t in prices.columns}

    benchmark_var = float(aligned["benchmark"].var())
    if benchmark_var <= 1e-12:
        return {t: 0.0 for t in prices.columns}

    betas = {}
    for ticker in prices.columns:
        cov = float(aligned[[ticker, "benchmark"]].cov().iloc[0, 1])
        betas[ticker] = cov / benchmark_var
    return betas


def serialize_history(price_frame):
    return {
        "dates": [idx.strftime("%Y-%m-%d") for idx in price_frame.index],
        "series": {
            col: [round(float(v), 6) for v in price_frame[col].tolist()]
            for col in price_frame.columns
        },
    }


def compute_realized_window_returns(prices, benchmark_prices=None, trading_days=30):
    if prices is None or len(prices) < 2:
        return None

    window_days = int(max(1, min(trading_days, len(prices) - 1)))
    window_prices = prices.iloc[-(window_days + 1):]

    asset_returns = (
        (window_prices.iloc[-1] / window_prices.iloc[0] - 1.0) * 100.0
    ).to_dict()
    benchmark_return_pct = None

    if benchmark_prices is not None and len(benchmark_prices) >= len(window_prices):
        benchmark_window = benchmark_prices.reindex(window_prices.index).dropna()
        if len(benchmark_window) >= 2:
            benchmark_return_pct = float(
                (benchmark_window.iloc[-1] / benchmark_window.iloc[0] - 1.0) * 100.0
            )

    return {
        "trading_days": window_days,
        "start": window_prices.index[0].strftime("%Y-%m-%d"),
        "end": window_prices.index[-1].strftime("%Y-%m-%d"),
        "asset_returns_pct": {
            ticker: round(float(value), 2)
            for ticker, value in asset_returns.items()
        },
        "benchmark_return_pct": (
            round(float(benchmark_return_pct), 2)
            if benchmark_return_pct is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Covariance & delta
# ---------------------------------------------------------------------------

def compute_covariance(prices):
    """Ledoit-Wolf shrinkage covariance on daily returns, annualized."""
    returns = prices.pct_change().dropna()
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * 252  # annualize
    return pd.DataFrame(cov, index=prices.columns, columns=prices.columns), returns


def compute_delta(returns, market_weights):
    """
    Risk-aversion coefficient: delta = (r_mkt - r_f) / sigma_mkt^2.
    Uses equal-weight proxy for market if weights unavailable.
    Falls back to 2.5.
    """
    try:
        w = np.array([market_weights.get(c, 0) for c in returns.columns])
        w = w / w.sum()
        mkt_ret = (returns.values @ w)
        excess = mkt_ret.mean() * 252
        var = mkt_ret.var() * 252
        if var > 1e-10:
            delta = excess / var
            if 0.5 < delta < 10:
                return delta
    except Exception:
        pass
    return 2.5


# ---------------------------------------------------------------------------
# View construction helpers
# ---------------------------------------------------------------------------

def build_P_Q(views, tickers):
    """
    Build the K x N picking matrix P and K x 1 view vector Q.

    Each view dict has:
        type: "absolute" | "relative"
        assets: list of tickers (1 for absolute, 2 for relative)
        return_pct: expected return in percent
        confidence_pct: 0-100 (used later for Omega)
    """
    N = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    P_rows = []
    Q_vals = []

    for v in views:
        row = np.zeros(N)
        assets = v["assets"]
        ret = v["return_pct"] / 100.0  # convert to decimal

        if v["type"] == "absolute":
            if assets[0] in ticker_idx:
                row[ticker_idx[assets[0]]] = 1.0
                P_rows.append(row)
                Q_vals.append(ret)
        else:  # relative: asset[0] outperforms asset[1] by return_pct
            if len(assets) >= 2 and assets[0] in ticker_idx and assets[1] in ticker_idx:
                row[ticker_idx[assets[0]]] = 1.0
                row[ticker_idx[assets[1]]] = -1.0
                P_rows.append(row)
                Q_vals.append(ret)

    P = np.array(P_rows) if P_rows else np.empty((0, N))
    Q = np.array(Q_vals).reshape(-1, 1) if Q_vals else np.empty((0, 1))
    return P, Q


# ---------------------------------------------------------------------------
# Idzorek's confidence-based Omega
# ---------------------------------------------------------------------------

def _idzorek_omega_element(k, P, Q, tau, Sigma, Pi, w_mkt, confidence_pct):
    """
    For view k, find omega_k such that the posterior weight tilt equals
    confidence_pct/100 of the 100%-confidence tilt.

    At 100% confidence (omega_k -> 0), the posterior is fully determined
    by the view. We solve for the omega_k that scales the weight tilt
    to match the desired confidence level.
    """
    conf = confidence_pct / 100.0
    conf = np.clip(conf, 0.01, 0.99)

    N = Sigma.shape[0]
    tau_Sigma = tau * Sigma
    tau_Sigma_inv = np.linalg.inv(tau_Sigma)

    p_k = P[k:k+1, :]  # 1 x N

    # 100% confidence posterior (omega_k = very small)
    omega_100 = np.eye(1) * 1e-10
    P_k = p_k
    Q_k = Q[k:k+1]
    M_100 = np.linalg.inv(tau_Sigma_inv + P_k.T @ np.linalg.inv(omega_100) @ P_k)
    E_100 = M_100 @ (tau_Sigma_inv @ Pi + P_k.T @ np.linalg.inv(omega_100) @ Q_k)

    # Optimal weights at 100% confidence
    try:
        from pypfopt import EfficientFrontier
        ef = EfficientFrontier(
            pd.Series(E_100.flatten()),
            pd.DataFrame(Sigma),
        )
        ef.max_sharpe(risk_free_rate=0.0)
        w_100 = np.array(list(ef.clean_weights().values()))
    except Exception:
        Sigma_inv = np.linalg.inv(Sigma)
        w_100 = Sigma_inv @ E_100.flatten()
        w_100 = w_100 / np.sum(np.abs(w_100))

    # Target tilt = conf * (w_100 - w_mkt)
    tilt_100 = w_100 - w_mkt
    target_tilt = conf * tilt_100

    # Binary search for omega_k
    lo, hi = 1e-12, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        omega_k = np.array([[mid]])
        M_k = np.linalg.inv(tau_Sigma_inv + P_k.T @ np.linalg.inv(omega_k) @ P_k)
        E_k = M_k @ (tau_Sigma_inv @ Pi + P_k.T @ np.linalg.inv(omega_k) @ Q_k)

        try:
            ef2 = EfficientFrontier(
                pd.Series(E_k.flatten()),
                pd.DataFrame(Sigma),
            )
            ef2.max_sharpe(risk_free_rate=0.0)
            w_k = np.array(list(ef2.clean_weights().values()))
        except Exception:
            Sigma_inv = np.linalg.inv(Sigma)
            w_k = Sigma_inv @ E_k.flatten()
            w_k = w_k / np.sum(np.abs(w_k))

        actual_tilt = w_k - w_mkt
        # Compare norm of actual tilt vs target tilt. Higher omega means less
        # confidence, so the posterior tilt shrinks as omega increases.
        if np.linalg.norm(actual_tilt) > np.linalg.norm(target_tilt):
            lo = mid  # omega too small (too confident), increase it
        else:
            hi = mid  # omega too large (not confident enough), decrease it

    omega_k_val = (lo + hi) / 2.0
    return max(omega_k_val, 1e-10)


def build_omega_idzorek(P, Q, tau, Sigma, Pi, w_mkt, confidence_pcts):
    """
    Build diagonal Omega matrix using Idzorek's method.
    confidence_pcts: list of confidence percentages (0-100) per view.
    """
    K = P.shape[0]
    omega_diag = np.zeros(K)
    for k in range(K):
        omega_diag[k] = _idzorek_omega_element(
            k, P, Q, tau, Sigma, Pi, w_mkt, confidence_pcts[k]
        )
    return np.diag(omega_diag), omega_diag


# ---------------------------------------------------------------------------
# Core BL math
# ---------------------------------------------------------------------------

def bl_posterior(tau, Sigma, Pi, P, Q, Omega):
    """
    Compute Black-Litterman posterior expected returns and covariance.

    E = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1 * Pi + P'*Omega^-1 * Q]
    Sigma_BL = Sigma + [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
    """
    tau_Sigma = tau * Sigma
    tau_Sigma_inv = np.linalg.inv(tau_Sigma)
    Omega_inv = np.linalg.inv(Omega)

    # Posterior precision
    M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
    M = np.linalg.inv(M_inv)

    # Posterior expected returns
    E = M @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

    # Posterior covariance
    Sigma_BL = Sigma + M

    return E.flatten(), Sigma_BL


def compute_view_portfolios(P, Q, tau, Sigma, Omega, Pi, w_mkt, tickers):
    """
    For each view k, compute:
    - The normalized view portfolio exposure vector
    - Lambda, a raw view-strength coefficient rather than a bounded weight
    """
    K = P.shape[0]
    N = len(tickers)

    decomposition = []
    for k in range(K):
        p_k = P[k:k+1, :]  # 1 x N
        q_k = Q[k:k+1]     # 1 x 1
        omega_k = Omega[k, k]

        # View portfolio: proportional to Sigma^-1 @ p_k'
        Sigma_inv = np.linalg.inv(Sigma)
        w_view = Sigma_inv @ p_k.T
        w_view = w_view.flatten()
        w_view_norm = w_view / np.sum(np.abs(w_view)) if np.sum(np.abs(w_view)) > 1e-10 else w_view

        # Lambda: the scalar weight of this view in the combined portfolio
        # lambda_k = (q_k - p_k @ Pi) / (omega_k + p_k @ tau_Sigma @ p_k')
        numerator = float(q_k - p_k @ Pi)
        denominator = float(omega_k + p_k @ (tau * Sigma) @ p_k.T)
        lam = numerator / denominator if abs(denominator) > 1e-10 else 0.0

        decomposition.append({
            "view_index": k,
            "weights": {tickers[i]: round(float(w_view_norm[i]), 4) for i in range(N)},
            "lambda": round(lam, 6),
        })

    return decomposition


def implied_views(P, Pi):
    """Reverse BL: Q_implied = P @ Pi."""
    Q_implied = P @ Pi
    return Q_implied.flatten()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_bl_analysis(tickers, market_weights, views, tau=0.05, risk_free_rate=0.02):
    """
    Full Black-Litterman analysis.

    Args:
        tickers: list of asset tickers
        market_weights: dict {ticker: weight} (normalized)
        views: list of view dicts with type, assets, return_pct, confidence_pct
        tau: scalar (default 0.05)
        risk_free_rate: float

    Returns dict with all BL outputs.
    """
    # 1. Fetch prices
    prices = fetch_prices(tickers)
    if prices is None:
        return {"error": "Not enough price data for the given tickers."}

    valid_tickers = list(prices.columns)
    N = len(valid_tickers)
    if N < 2:
        return {"error": "Need at least 2 tickers with sufficient price history."}

    benchmark_ticker = select_market_benchmark(valid_tickers)
    benchmark_prices = fetch_benchmark_prices(benchmark_ticker)
    if benchmark_prices is not None:
        aligned_prices = prices.join(benchmark_prices.rename(benchmark_ticker), how="inner").dropna()
        if len(aligned_prices) > 60:
            prices = aligned_prices[valid_tickers]
            benchmark_prices = aligned_prices[benchmark_ticker]
        else:
            benchmark_prices = None

    # 2. Covariance matrix (Ledoit-Wolf)
    Sigma_df, returns = compute_covariance(prices)
    Sigma = Sigma_df.values
    betas = compute_asset_betas(prices, benchmark_prices) if benchmark_prices is not None else {
        t: 0.0 for t in valid_tickers
    }
    sample_window = {
        "start": prices.index.min().strftime("%Y-%m-%d"),
        "end": prices.index.max().strftime("%Y-%m-%d"),
        "trading_days": max(len(prices) - 1, 0),
    }

    # 3. Market weights vector
    w_mkt = np.array([market_weights.get(t, 1.0 / N) for t in valid_tickers])
    w_mkt = w_mkt / w_mkt.sum()

    # 4. Delta from market data
    delta = compute_delta(returns, {t: market_weights.get(t, 1.0/N) for t in valid_tickers})

    # 5. Implied equilibrium returns: Pi = delta * Sigma * w_mkt
    Pi = delta * Sigma @ w_mkt  # N-vector

    # 6. Build P and Q from views
    P, Q = build_P_Q(views, valid_tickers)
    K = P.shape[0]

    if K == 0:
        return {"error": "No valid views could be constructed from the inputs."}

    # 7. Confidence percentages for Omega
    confidence_pcts = []
    view_idx = 0
    for v in views:
        assets = v["assets"]
        if v["type"] == "absolute" and assets[0] in valid_tickers:
            confidence_pcts.append(v.get("confidence_pct", 50))
            view_idx += 1
        elif v["type"] == "relative" and len(assets) >= 2 and assets[0] in valid_tickers and assets[1] in valid_tickers:
            confidence_pcts.append(v.get("confidence_pct", 50))
            view_idx += 1

    # 8. Build Omega via Idzorek's method
    Omega, omega_diag = build_omega_idzorek(P, Q, tau, Sigma, Pi.reshape(-1, 1), w_mkt, confidence_pcts)

    # 9. BL posterior
    E, Sigma_BL = bl_posterior(tau, Sigma, Pi.reshape(-1, 1), P, Q, Omega)

    # 10. MVO optimal weights via PyPortfolioOpt
    from pypfopt import EfficientFrontier
    try:
        ef = EfficientFrontier(
            pd.Series(E, index=valid_tickers),
            pd.DataFrame(Sigma_BL, index=valid_tickers, columns=valid_tickers),
        )
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        raw_weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        bl_weights = {k: round(float(v), 4) for k, v in raw_weights.items()}
        expected_return_pf = round(float(perf[0]) * 100, 2)
        volatility_pf = round(float(perf[1]) * 100, 2)
        sharpe_pf = round(float(perf[2]), 3)
    except Exception:
        # Fallback: analytical MVO
        Sigma_inv = np.linalg.inv(Sigma_BL)
        w_opt = Sigma_inv @ E
        w_opt = w_opt / np.sum(np.abs(w_opt))
        w_opt = np.maximum(w_opt, 0)
        w_opt = w_opt / w_opt.sum()
        bl_weights = {valid_tickers[i]: round(float(w_opt[i]), 4) for i in range(N)}
        port_ret = float(w_opt @ E)
        port_vol = float(np.sqrt(w_opt @ Sigma_BL @ w_opt))
        expected_return_pf = round(port_ret * 100, 2)
        volatility_pf = round(port_vol * 100, 2)
        sharpe_pf = round((port_ret - risk_free_rate) / port_vol, 3) if port_vol > 1e-10 else 0.0

    # 11. Equilibrium weights (from market weights)
    eq_weights = {valid_tickers[i]: round(float(w_mkt[i]), 4) for i in range(N)}

    # 12. View portfolio decomposition
    view_decomp = compute_view_portfolios(P, Q, tau, Sigma, Omega, Pi.reshape(-1, 1), w_mkt, valid_tickers)

    # 13. Implied views (reverse BL)
    Q_impl = implied_views(P, Pi)
    implied_views_list = []
    view_j = 0
    for v in views:
        assets = v["assets"]
        if v["type"] == "absolute" and assets[0] in valid_tickers:
            implied_views_list.append({
                "view_index": view_j,
                "type": v["type"],
                "assets": assets,
                "stated_return_pct": v["return_pct"],
                "implied_return_pct": round(float(Q_impl[view_j]) * 100, 2),
            })
            view_j += 1
        elif v["type"] == "relative" and len(assets) >= 2 and assets[0] in valid_tickers and assets[1] in valid_tickers:
            implied_views_list.append({
                "view_index": view_j,
                "type": v["type"],
                "assets": assets,
                "stated_return_pct": v["return_pct"],
                "implied_return_pct": round(float(Q_impl[view_j]) * 100, 2),
            })
            view_j += 1

    # 14. Market portfolio baseline stats
    mkt_ret = float(w_mkt @ E)
    mkt_vol = float(np.sqrt(w_mkt @ Sigma_BL @ w_mkt))
    mkt_sharpe = round((mkt_ret - risk_free_rate) / mkt_vol, 3) if mkt_vol > 1e-10 else 0.0

    # 15. Build return table
    default_realized_window = compute_realized_window_returns(
        prices,
        benchmark_prices=benchmark_prices,
        trading_days=30,
    )
    return_table = []
    for i, t in enumerate(valid_tickers):
        realized_return_pct = None
        excess_vs_benchmark_pct = None
        benchmark_return_pct = None
        if default_realized_window is not None:
            realized_return_pct = default_realized_window["asset_returns_pct"].get(t)
            benchmark_return_pct = default_realized_window.get("benchmark_return_pct")
            if realized_return_pct is not None and benchmark_return_pct is not None:
                excess_vs_benchmark_pct = round(
                    float(realized_return_pct - benchmark_return_pct),
                    2,
                )

        return_table.append({
            "ticker": t,
            "beta_vs_benchmark": round(float(betas.get(t, 0.0)), 3),
            "equilibrium_return_pct": round(float(Pi[i]) * 100, 2),
            "posterior_return_pct": round(float(E[i]) * 100, 2),
            "market_weight": eq_weights.get(t, 0.0),
            "bl_weight": bl_weights.get(t, 0.0),
            "weight_tilt": round(float(bl_weights.get(t, 0.0) - eq_weights.get(t, 0.0)), 4),
            "realized_return_pct": realized_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "excess_vs_benchmark_pct": excess_vs_benchmark_pct,
        })

    return {
        "equilibrium_weights": eq_weights,
        "bl_weights": bl_weights,
        "posterior_returns": {valid_tickers[i]: round(float(E[i]) * 100, 2) for i in range(N)},
        "equilibrium_returns": {valid_tickers[i]: round(float(Pi[i]) * 100, 2) for i in range(N)},
        "betas": {t: round(float(betas.get(t, 0.0)), 3) for t in valid_tickers},
        "benchmark": {
            "ticker": benchmark_ticker,
            "label": BENCHMARK_LABELS.get(benchmark_ticker, benchmark_ticker),
        },
        "benchmark_window": sample_window,
        "realized_window": default_realized_window,
        "price_history": serialize_history(prices),
        "benchmark_history": (
            serialize_history(benchmark_prices.to_frame(name=benchmark_ticker))
            if benchmark_prices is not None else None
        ),
        "return_table": return_table,
        "view_decomposition": view_decomp,
        "implied_views": implied_views_list,
        "omega_diagonal": [round(float(x), 8) for x in omega_diag],
        "portfolio_stats": {
            "expected_return": expected_return_pf,
            "volatility": volatility_pf,
            "sharpe_ratio": sharpe_pf,
        },
        "market_stats": {
            "expected_return": round(mkt_ret * 100, 2),
            "volatility": round(mkt_vol * 100, 2),
            "sharpe_ratio": mkt_sharpe,
        },
        "delta": round(delta, 4),
        "tau": tau,
        "risk_free_rate": risk_free_rate,
        "tickers": valid_tickers,
        "error": None,
    }
