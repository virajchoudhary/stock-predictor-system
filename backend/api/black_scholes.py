"""
Black–Scholes Options Pricing Engine — v2
==========================================
Comprehensive engine supporting Indian (NSE/BSE) and US markets.

Model: Black–Scholes–Merton with continuous dividend yield q
    d₁ = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
    d₂ = d₁ − σ·√T
    Call = S·e^{-qT}·N(d₁) − K·e^{-rT}·N(d₂)
    Put  = K·e^{-rT}·N(−d₂) − S·e^{-qT}·N(−d₁)

References:
  - Black, Fischer; Scholes, Myron (1973). Journal of Political Economy.
  - Hull, John C. "Options, Futures, and Other Derivatives" (11th ed.)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ──────────────────────────────────────────────────────────────────────
# Market Detection & Defaults
# ──────────────────────────────────────────────────────────────────────

INDIA_SUFFIXES = {".NS", ".BO", ".BSE"}
INDIA_PREFIXES = {"NIFTY", "BANKNIFTY", "SENSEX", "^NSEI", "^NSEBANK"}

# Default risk-free rates (annualised, as of 2024-2025)
DEFAULT_RATES = {
    "IN": 0.068,   # India 10-yr Govt Bond yield ~6.8%
    "US": 0.053,   # US 10-yr Treasury yield ~5.3%
}

# Default market volatility estimates (σ) when live data is unavailable
DEFAULT_VOLATILITY = {
    "IN": 0.20,    # Indian equities ~20% historical vol
    "US": 0.25,    # US equities ~25% historical vol
}


def detect_market(symbol: str) -> str:
    """
    Auto-detect market from ticker symbol.
    Returns 'IN' for India, 'US' for United States.
    """
    sym_upper = symbol.upper().strip()
    for prefix in INDIA_PREFIXES:
        if sym_upper.startswith(prefix):
            return "IN"
    for suffix in INDIA_SUFFIXES:
        if sym_upper.endswith(suffix):
            return "IN"
    return "US"


def get_currency_symbol(market: str) -> str:
    return "₹" if market == "IN" else "$"


def get_default_rate(market: str) -> float:
    return DEFAULT_RATES.get(market, 0.05)


def format_price(value: float, market: str, decimals: int = 2) -> str:
    sym = get_currency_symbol(market)
    return f"{sym}{value:,.{decimals}f}"


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────

def _validate_inputs(S: float, K: float, T: float, r: float,
                     sigma: float, q: float) -> None:
    if S <= 0:
        raise ValueError(f"Spot price S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be > 0, got {K}")
    if sigma < 0:
        raise ValueError(f"Volatility sigma must be >= 0, got {sigma}")
    if T < 0:
        raise ValueError(f"Time to expiry T must be >= 0, got {T}")


# ──────────────────────────────────────────────────────────────────────
# Core d1, d2
# ──────────────────────────────────────────────────────────────────────

def compute_d1(S: float, K: float, T: float, r: float,
               sigma: float, q: float = 0.0) -> float:
    """Black–Scholes d₁."""
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def compute_d2(S: float, K: float, T: float, r: float,
               sigma: float, q: float = 0.0) -> float:
    """Black–Scholes d₂ = d₁ − σ√T."""
    return compute_d1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)


# ──────────────────────────────────────────────────────────────────────
# Option Pricing
# ──────────────────────────────────────────────────────────────────────

def bs_call_price(S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0.0) -> float:
    """Theoretical European call price (BSM model)."""
    _validate_inputs(S, K, T, r, sigma, q)
    if T <= 1e-10:
        return max(S - K, 0.0)
    if sigma <= 1e-10:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    return max(S * math.exp(-q * T) * norm.cdf(d1)
               - K * math.exp(-r * T) * norm.cdf(d2), 0.0)


def bs_put_price(S: float, K: float, T: float, r: float,
                 sigma: float, q: float = 0.0) -> float:
    """Theoretical European put price (BSM model)."""
    _validate_inputs(S, K, T, r, sigma, q)
    if T <= 1e-10:
        return max(K - S, 0.0)
    if sigma <= 1e-10:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    return max(K * math.exp(-r * T) * norm.cdf(-d2)
               - S * math.exp(-q * T) * norm.cdf(-d1), 0.0)


def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, q: float = 0.0,
             option_type: str = "call") -> float:
    if option_type.lower() == "call":
        return bs_call_price(S, K, T, r, sigma, q)
    elif option_type.lower() == "put":
        return bs_put_price(S, K, T, r, sigma, q)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ──────────────────────────────────────────────────────────────────────
# Greeks
# ──────────────────────────────────────────────────────────────────────

def greeks(S: float, K: float, T: float, r: float,
           sigma: float, q: float = 0.0) -> Dict[str, Dict[str, float]]:
    """
    Compute Δ, Γ, Θ, ν, ρ for both call and put.
    Theta is per calendar day; Vega and Rho are per 1% change.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    if T <= 1e-10 or sigma <= 1e-10:
        itm = 1.0 if S > K else (0.5 if abs(S - K) < 1e-10 else 0.0)
        return {
            "call": {"delta": itm, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0},
            "put":  {"delta": itm - 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0},
        }

    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)
    exp_qT = math.exp(-q * T)
    exp_rT = math.exp(-r * T)
    n_d1 = norm.pdf(d1)

    # Delta
    call_delta = exp_qT * norm.cdf(d1)
    put_delta = -exp_qT * norm.cdf(-d1)

    # Gamma (identical for call and put)
    gamma_val = (exp_qT * n_d1) / (S * sigma * sqrt_T)

    # Theta (per calendar day)
    common = -(S * sigma * exp_qT * n_d1) / (2 * sqrt_T)
    call_theta = (common - r * K * exp_rT * norm.cdf(d2)
                  + q * S * exp_qT * norm.cdf(d1)) / 365.0
    put_theta = (common + r * K * exp_rT * norm.cdf(-d2)
                 - q * S * exp_qT * norm.cdf(-d1)) / 365.0

    # Vega (per 1% σ change)
    vega_val = (S * exp_qT * n_d1 * sqrt_T) / 100.0

    # Rho (per 1% rate change)
    call_rho = (K * T * exp_rT * norm.cdf(d2)) / 100.0
    put_rho = -(K * T * exp_rT * norm.cdf(-d2)) / 100.0

    return {
        "call": {
            "delta": round(call_delta, 6), "gamma": round(gamma_val, 6),
            "theta": round(call_theta, 6), "vega": round(vega_val, 6),
            "rho": round(call_rho, 6),
        },
        "put": {
            "delta": round(put_delta, 6), "gamma": round(gamma_val, 6),
            "theta": round(put_theta, 6), "vega": round(vega_val, 6),
            "rho": round(put_rho, 6),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Implied Volatility
# ──────────────────────────────────────────────────────────────────────

def implied_volatility(market_price: float, S: float, K: float, T: float,
                       r: float, q: float = 0.0, option_type: str = "call",
                       tol: float = 1e-8, max_iter: int = 200) -> Optional[float]:
    """
    Solve for IV using Brent's method. Returns None if solver fails.
    """
    if market_price <= 0 or T <= 1e-10:
        return None

    intrinsic = (max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
                 if option_type.lower() == "call"
                 else max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0))

    if market_price < intrinsic - 1e-6:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=tol, maxiter=max_iter)
        return round(iv, 6)
    except (ValueError, RuntimeError):
        return None


# ──────────────────────────────────────────────────────────────────────
# Confidence Score
# ──────────────────────────────────────────────────────────────────────

def confidence_score(S: float, K: float, T: float, sigma: float,
                     market_price: Optional[float] = None,
                     theoretical_price: Optional[float] = None) -> Dict:
    """
    Compute a 0–100 confidence score for a trading signal.

    Factors:
      - Time value: more time = more confident (up to 40pts)
      - Moneyness: near ATM = higher confidence (up to 30pts)
      - Mispricing magnitude: larger gap = higher confidence (up to 30pts)
    """
    # Time factor: penalise options with < 7 days left
    time_score = min(T * 365 / 60.0, 1.0) * 40.0   # max at 60+ days

    # Moneyness factor: ATM options have most reliable BS pricing
    moneyness = abs(math.log(S / K))
    money_score = max(0.0, 1.0 - moneyness / 0.3) * 30.0  # decays beyond 30% OTM/ITM

    # Mispricing magnitude factor
    if market_price and theoretical_price and theoretical_price > 0:
        deviation = abs(market_price - theoretical_price) / theoretical_price
        misprice_score = min(deviation / 0.20, 1.0) * 30.0   # max at 20%+ deviation
    else:
        misprice_score = 0.0

    total = round(time_score + money_score + misprice_score, 1)

    if total >= 70:
        level = "High"
    elif total >= 40:
        level = "Medium"
    else:
        level = "Low"

    return {
        "score": total,
        "level": level,
        "time_score": round(time_score, 1),
        "moneyness_score": round(money_score, 1),
        "mispricing_score": round(misprice_score, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Trading Signal
# ──────────────────────────────────────────────────────────────────────

def valuation_signal(market_price: float, theoretical_price: float,
                     threshold_pct: float = 5.0,
                     currency_symbol: str = "$") -> Dict[str, str]:
    """
    Generate BUY / SELL / HOLD signal with currency-aware reasoning.
    """
    if theoretical_price <= 0:
        return {"signal": "HOLD",
                "reasoning": "Theoretical price is near zero (deep OTM). Avoid trading.",
                "deviation_pct": 0.0}

    deviation_pct = ((market_price - theoretical_price) / theoretical_price) * 100.0

    p_mkt = f"{currency_symbol}{market_price:.2f}"
    p_theo = f"{currency_symbol}{theoretical_price:.2f}"

    if deviation_pct < -threshold_pct:
        return {
            "signal": "BUY",
            "reasoning": (
                f"Market price ({p_mkt}) is {abs(deviation_pct):.1f}% below "
                f"theoretical value ({p_theo}). Option appears undervalued."
            ),
            "deviation_pct": round(deviation_pct, 2),
        }
    elif deviation_pct > threshold_pct:
        return {
            "signal": "SELL",
            "reasoning": (
                f"Market price ({p_mkt}) is {deviation_pct:.1f}% above "
                f"theoretical value ({p_theo}). Option appears overvalued — "
                f"consider writing/selling."
            ),
            "deviation_pct": round(deviation_pct, 2),
        }
    else:
        return {
            "signal": "HOLD",
            "reasoning": (
                f"Market price ({p_mkt}) is within {threshold_pct:.0f}% of "
                f"theoretical value ({p_theo}). Option is fairly priced."
            ),
            "deviation_pct": round(deviation_pct, 2),
        }


# ──────────────────────────────────────────────────────────────────────
# Historical Volatility from price series
# ──────────────────────────────────────────────────────────────────────

def historical_volatility(prices: list, window: int = 30) -> float:
    """
    Compute annualised historical volatility from a daily closing price series.
    Uses log returns, annualised by √252.
    """
    if len(prices) < 2:
        return 0.20  # fallback
    log_returns = [math.log(prices[i] / prices[i - 1])
                   for i in range(1, len(prices)) if prices[i - 1] > 0]
    if len(log_returns) < 2:
        return 0.20
    series = log_returns[-window:]
    mean_r = sum(series) / len(series)
    variance = sum((r - mean_r) ** 2 for r in series) / (len(series) - 1)
    return round(math.sqrt(variance) * math.sqrt(252), 4)


# ──────────────────────────────────────────────────────────────────────
# Top-level analysis function
# ──────────────────────────────────────────────────────────────────────

def analyze_option(S: float, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0,
                   market_price: Optional[float] = None,
                   option_type: str = "call",
                   signal_threshold: float = 5.0,
                   market: str = "US") -> Dict:
    """
    Complete Black–Scholes analysis for one option contract.

    Parameters
    ----------
    market : 'IN' or 'US' — determines currency symbol in signal text
    """
    currency = get_currency_symbol(market)

    call_price = bs_call_price(S, K, T, r, sigma, q)
    put_price = bs_put_price(S, K, T, r, sigma, q)
    option_greeks = greeks(S, K, T, r, sigma, q)

    # Put-call parity check
    parity_lhs = call_price - put_price
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    parity_error = abs(parity_lhs - parity_rhs)

    result = {
        "market": market,
        "currency": currency,
        "inputs": {
            "spot_price": S, "strike_price": K,
            "time_to_expiry_years": T, "risk_free_rate": r,
            "volatility": sigma, "dividend_yield": q,
            "option_type": option_type,
        },
        "theoretical_price": {
            "call": round(call_price, 4),
            "put": round(put_price, 4),
        },
        "greeks": option_greeks,
        "put_call_parity": {
            "C_minus_P": round(parity_lhs, 4),
            "S_e_qT_minus_K_e_rT": round(parity_rhs, 4),
            "error": round(parity_error, 8),
            "holds": parity_error < 0.01,
        },
    }

    theo = call_price if option_type.lower() == "call" else put_price

    if market_price is not None and market_price > 0:
        iv = implied_volatility(market_price, S, K, T, r, q, option_type)
        signal = valuation_signal(market_price, theo, signal_threshold, currency)
        conf = confidence_score(S, K, T, sigma, market_price, theo)

        result["market_price"] = round(market_price, 4)
        result["implied_volatility"] = round(iv, 6) if iv is not None else None
        result["valuation"] = signal
        result["confidence"] = conf
    else:
        # Confidence without market price comparison
        result["confidence"] = confidence_score(S, K, T, sigma)

    return result
