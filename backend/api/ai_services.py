import os
import json
import numpy as np
import pandas as pd
from groq import Groq
import yfinance as yf

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

try:
    from fastembed import TextEmbedding
    embedding_model = None  # disabled to skip download
except Exception:
    embedding_model = None

import riskfolio as rp
import ta

from polygon import RESTClient
polygon_api_key = os.environ.get("POLYGON_API_KEY")
polygon_client  = RESTClient(api_key=polygon_api_key) if polygon_api_key else None


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
api_key     = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None


class GroqService:
    @staticmethod
    def chat(message):
        if not groq_client:
            return "Groq API Key not found."
        try:
            messages = message if isinstance(message, list) else [
                {"role": "system", "content": "You are a helpful financial assistant for QuantVision."},
                {"role": "user",   "content": message},
            ]
            resp = groq_client.chat.completions.create(
                messages=messages, model="llama-3.3-70b-versatile"
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def analyze_allocation(tickers, allocation, risk_tolerance):
        if not groq_client:
            return "Groq API Key not found."
        try:
            prompt = f"""
            You are a portfolio manager. Analyze this allocation:
            Risk Tolerance: {risk_tolerance} (0=Conservative, 1=Aggressive)
            Allocation: {allocation}
            Explain WHY this allocation makes sense. Keep it concise (3-4 sentences).
            """
            resp = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a financial expert."},
                    {"role": "user",   "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Could not generate analysis: {str(e)}"

    @staticmethod
    def score_news_sentiment(ticker, headlines):
        """Returns float in [-1, +1]. Uses Groq to score headlines."""
        if not groq_client or not headlines:
            return 0.0
        try:
            joined = "\n".join(f"- {h}" for h in headlines[:5])
            prompt = f"""
You are a financial news analyst. Given these recent headlines for {ticker},
return ONLY a single float between -1.0 and 1.0 (-1=very bearish, 0=neutral, +1=very bullish).
Return only the number, nothing else.
Headlines:
{joined}
"""
            resp = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=10,
            )
            return float(np.clip(float(resp.choices[0].message.content.strip()), -1.0, 1.0))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Sector risk multipliers (higher = riskier)
# ---------------------------------------------------------------------------
SECTOR_RISK = {
    "technology": 1.4, "semiconductors": 1.5, "crypto": 2.0,
    "energy": 1.3, "consumer cyclical": 1.3, "communication services": 1.2,
    "financial services": 1.1, "industrials": 1.1, "real estate": 1.2,
    "basic materials": 1.2, "healthcare": 0.9, "consumer defensive": 0.8,
    "utilities": 0.7, "computers": 1.4, "oil": 1.3, "banks": 1.1,
    "pharma": 0.9, "auto": 1.2,
}

def get_sector_risk(sector_str):
    if not sector_str:
        return 1.0
    s = sector_str.lower()
    for key, mult in SECTOR_RISK.items():
        if key in s:
            return mult
    return 1.0


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

def get_price_history(symbol, period="6mo"):
    if polygon_client:
        try:
            from datetime import datetime, timedelta
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
            days = period_days.get(period, 180)
            end_date   = datetime.now()
            start_date = end_date - timedelta(days=days)
            aggs = polygon_client.get_aggs(
                ticker=symbol.upper(), multiplier=1, timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"), limit=50000,
            )
            if aggs:
                df = pd.DataFrame([{
                    "Open": a.open, "High": a.high,
                    "Low": a.low, "Close": a.close, "Volume": a.volume,
                } for a in aggs],
                index=pd.to_datetime([a.timestamp for a in aggs], unit="ms"))
                df.index.name = "Date"
                if not df.empty:
                    return df
        except Exception:
            pass
    return yf.Ticker(symbol).history(period=period)


def get_ticker_info(symbol):
    if polygon_client:
        try:
            detail = polygon_client.get_ticker_details(symbol.upper())
            if detail:
                return {
                    "marketCap": getattr(detail, "market_cap", "N/A"),
                    "currentPrice": "N/A", "trailingEps": "N/A",
                    "trailingPE": "N/A", "debtToEquity": "N/A",
                    "totalDebt": "N/A", "revenueGrowth": "N/A",
                    "sector": getattr(detail, "sic_description", "N/A"),
                    "name": getattr(detail, "name", symbol),
                }
        except Exception:
            pass
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Fundamental Score [0, 1]
# ---------------------------------------------------------------------------

def compute_fundamental_score(info):
    scores = []
    for val, scale, invert in [
        (info.get("trailingPE"),    50.0,  True),
        (info.get("debtToEquity"),  200.0, True),
        (info.get("revenueGrowth"), 0.20,  False),
    ]:
        try:
            v = float(val)
            if invert:
                scores.append(float(np.clip(1.0 - v / scale, 0.0, 1.0)))
            else:
                scores.append(float(np.clip(v / scale, 0.0, 1.0)))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Pre-fetch fundamentals + sentiment per ticker (once before training)
# ---------------------------------------------------------------------------

def fetch_ticker_features(tickers):
    features = {}
    for t in tickers:
        try:
            info   = get_ticker_info(t)
            sector = info.get("sector", info.get("sic_description", ""))
            f_score = compute_fundamental_score(info)
            headlines = []
            try:
                raw = yf.Ticker(t).news or []
                for n in raw[:5]:
                    c = n.get("content", n)
                    title = c.get("title", n.get("title", ""))
                    if title:
                        headlines.append(title)
            except Exception:
                pass
            sentiment   = GroqService.score_news_sentiment(t, headlines)
            sector_risk = get_sector_risk(sector)
            features[t] = {
                "fundamental_score": f_score,
                "news_sentiment":    sentiment,
                "sector_risk":       sector_risk,
            }
        except Exception:
            features[t] = {"fundamental_score": 0.5, "news_sentiment": 0.0, "sector_risk": 1.0}
    return features


# ---------------------------------------------------------------------------
# Rolling technical indicators (fast numpy/pandas implementations)
# ---------------------------------------------------------------------------

def _rolling_rsi(prices, window=14):
    if len(prices) < window + 1:
        return 0.5
    deltas = np.diff(prices)
    gains  = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_g  = np.mean(gains[-window:])
    avg_l  = np.mean(losses[-window:])
    if avg_l < 1e-10:
        return 1.0
    return float(1.0 - 1.0 / (1.0 + avg_g / avg_l))

def _rolling_macd(prices):
    if len(prices) < 26:
        return 0.0
    s  = pd.Series(prices)
    e12 = s.ewm(span=12, adjust=False).mean()
    e26 = s.ewm(span=26, adjust=False).mean()
    macd   = e12 - e26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = float((macd - signal).iloc[-1])
    return float(np.clip(hist / (np.std(prices[-26:]) + 1e-8), -1.0, 1.0))

def _bollinger_pct(prices, window=20):
    if len(prices) < window:
        return 0.5
    s   = pd.Series(prices[-window:])
    mid = s.mean()
    std = s.std() + 1e-8
    return float(np.clip((prices[-1] - (mid - 2 * std)) / (4 * std), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Trend Predictor (unchanged from previous version)
# ---------------------------------------------------------------------------

class TrendPredictor:
    @staticmethod
    def get_peers(symbol, sector=None):
        is_indian = ".NS" in symbol or ".BO" in symbol
        if is_indian:
            peers = ["NIFTYBEES.NS", "GOLDBEES.NS"]
            if sector:
                sec = sector.lower()
                if "energy" in sec or "oil" in sec:       peers = ["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS"]
                elif "technology" in sec or "computers" in sec: peers = ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS"]
                elif "financial" in sec or "bank" in sec: peers = ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS"]
                elif "auto" in sec:                        peers = ["TATAMOTORS.NS","M&M.NS","MARUTI.NS"]
                elif "healthcare" in sec or "pharma" in sec: peers = ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"]
                elif "consumer" in sec:                    peers = ["HINDUNILVR.NS","ITC.NS","TITAN.NS"]
            return [p for p in peers if p != symbol]
        if "AAPL" in symbol: return ["MSFT", "GOOG", "AMZN"]
        if "TSLA" in symbol: return ["F", "GM", "RIVN"]
        return ["SPY", "QQQ"]

    @staticmethod
    def _parse_news(raw_news):
        parsed = []
        for n in raw_news:
            c = n.get("content", n)
            parsed.append({
                "title":     c.get("title",    n.get("title",     "No Title")),
                "publisher": (c.get("provider", {}).get("displayName") or n.get("publisher", "Unknown")),
                "link":      (c.get("canonicalUrl", {}).get("url") or n.get("link", "#")),
            })
        return parsed

    @staticmethod
    def predict(symbol):
        try:
            hist_daily  = get_price_history(symbol, period="6mo")
            hist_weekly = get_price_history(symbol, period="1y")
            if hist_weekly is not None and not hist_weekly.empty:
                hist_weekly = hist_weekly.resample("W").last().tail(30)
            if hist_daily is None or hist_daily.empty:
                return {"error": f"No data found for {symbol}."}
            info   = get_ticker_info(symbol)
            sector = info.get("sector", info.get("sic_description", ""))

            def extract_financials(d):
                if not d: return {}
                return {
                    "Market Cap":     d.get("marketCap", "N/A"),
                    "Current Price":  d.get("currentPrice", d.get("regularMarketPrice", "N/A")),
                    "EPS (Trailing)": d.get("trailingEps", "N/A"),
                    "PE Ratio":       d.get("trailingPE", "N/A"),
                    "Debt to Equity": d.get("debtToEquity", "N/A"),
                    "Total Debt":     d.get("totalDebt", "N/A"),
                    "Revenue Growth": d.get("revenueGrowth", "N/A"),
                    "Sector":         d.get("sector", d.get("sic_description", "N/A")),
                }

            financials = extract_financials(info)
            try:
                raw_news = yf.Ticker(symbol).news[:3] or []
                news     = TrendPredictor._parse_news(raw_news)
            except Exception:
                news = []
            news_text = "\n".join([f"- {n['title']} ({n['publisher']})" for n in news])
            peers = TrendPredictor.get_peers(symbol, sector)
            peer_data, peer_history = [], {}
            fd = financials.copy(); fd["Symbol"] = symbol
            peer_data.append(fd)
            peer_history[symbol] = hist_daily["Close"].tolist()
            for p in peers:
                try:
                    ph = get_price_history(p, period="6mo")
                    pi = get_ticker_info(p)
                    pf = extract_financials(pi); pf["Symbol"] = p
                    peer_data.append(pf)
                    if ph is not None and not ph.empty:
                        peer_history[p] = ph["Close"].tolist()
                except Exception:
                    continue

            if len(hist_daily) < 50:
                return {"error": "Not enough historical data."}
            close = hist_daily["Close"]; high = hist_daily["High"]
            low   = hist_daily["Low"];   volume = hist_daily["Volume"]

            current_rsi = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]) \
                if not pd.isna(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]) else 50.0
            macd_obj       = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            current_macd   = float(macd_obj.macd().iloc[-1])
            current_signal = float(macd_obj.macd_signal().iloc[-1])
            bb        = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_upper  = float(bb.bollinger_hband().iloc[-1])
            bb_lower  = float(bb.bollinger_lband().iloc[-1])
            bb_mid    = float(bb.bollinger_mavg().iloc[-1])
            current_atr = float(ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1])
            sma_50  = float(ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])
            sma_200_val = ta.trend.SMAIndicator(close=close, window=200).sma_indicator().iloc[-1]
            sma_200 = float(sma_200_val) if not pd.isna(sma_200_val) else None
            recent_close = float(close.iloc[-1])

            signals = []
            if current_rsi < 30:                   signals.append("Oversold (Bullish)")
            if current_rsi > 70:                   signals.append("Overbought (Bearish)")
            if current_macd > current_signal:      signals.append("MACD Bullish Crossover")
            if current_macd < current_signal:      signals.append("MACD Bearish Crossover")
            if recent_close > sma_50:              signals.append("Price > SMA50 (Bullish)")
            if sma_200 and recent_close > sma_200: signals.append("Price > SMA200 (Bullish)")
            if recent_close > bb_upper:            signals.append("Above Bollinger Upper (Overbought)")
            if recent_close < bb_lower:            signals.append("Below Bollinger Lower (Oversold)")

            bv = sum(1 for s in signals if "Bullish" in s or "Oversold" in s)
            be = sum(1 for s in signals if "Bearish" in s or "Overbought" in s)
            if bv > be:     trend = "UP";      change_pct = np.random.uniform(0.01, 0.05)
            elif be > bv:   trend = "DOWN";    change_pct = np.random.uniform(-0.05, -0.01)
            else:           trend = "NEUTRAL"; change_pct = np.random.uniform(-0.005, 0.005)
            predicted_price = recent_close * (1 + change_pct)

            prompt = f"""Analyze {symbol} (Sector: {sector}).
TECHNICAL: Price={recent_close}, RSI={current_rsi:.2f}, MACD={current_macd:.4f},
Signal={current_signal:.4f}, SMA50={sma_50:.2f}, SMA200={f'{sma_200:.2f}' if sma_200 else 'N/A'},
BB_Upper={bb_upper:.2f}, BB_Lower={bb_lower:.2f}, ATR={current_atr:.2f},
Signals={', '.join(signals)}, Trend={trend}
FUNDAMENTAL: {financials}
NEWS: {news_text}
PEERS: {peer_data}
Provide comprehensive analysis. Reference Bollinger Bands and ATR for volatility."""
            ai_reasoning = GroqService.chat(prompt)

            return {
                "symbol": symbol, "current_price": float(recent_close),
                "predicted_price": float(predicted_price), "trend": trend,
                "financials": financials, "news": news,
                "peer_financials": peer_data, "peer_history": peer_history,
                "weekly_data": hist_weekly["Close"].tolist() if hist_weekly is not None and not hist_weekly.empty else [],
                "technicals": {
                    "rsi": float(current_rsi), "macd": float(current_macd),
                    "signal": float(current_signal), "sma_50": float(sma_50),
                    "sma_200": float(sma_200) if sma_200 else None,
                    "bb_upper": float(bb_upper), "bb_lower": float(bb_lower),
                    "bb_mid": float(bb_mid), "atr": float(current_atr), "signals": signals,
                },
                "ohlc_data": json.loads(hist_daily.to_json(orient="split", date_format="iso")),
                "reasoning": ai_reasoning,
            }
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    @staticmethod
    def optimize(tickers, risk_tolerance):
        try:
            frames = {}
            for t in tickers:
                hist = get_price_history(t, period="2y")
                if hist is not None and not hist.empty:
                    frames[t] = hist["Close"]
            if len(frames) < 2:
                raise ValueError("Need at least 2 tickers.")
            prices  = pd.DataFrame(frames).dropna()
            if len(prices) < 60:
                raise ValueError("Not enough data.")
            returns = prices.pct_change().dropna()

            if risk_tolerance < 0.34:
                hrp = rp.HCPortfolio(returns=returns)
                w   = hrp.optimization(model="HRP", codependence="pearson",
                                       rm="MV", rf=0, linkage="ward", max_k=10, leaf_order=True)
                weights = w["weights"].to_dict()
            elif risk_tolerance < 0.67:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                w = port.optimization(model="Classic", rm="CVaR", obj="MinRisk", rf=0.05, l=0, hist=True)
                weights = w["weights"].to_dict()
            else:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                w = port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=0.05, l=0, hist=True)
                weights = w["weights"].to_dict()
            return {k: round(float(v), 4) for k, v in weights.items() if float(v) > 0.001}
        except Exception:
            n = len(tickers)
            return {t: round(1.0 / n, 4) for t in tickers}


# ============================================================================
# REINFORCEMENT LEARNING — INDUSTRY-GRADE ENSEMBLE
# ============================================================================
# Research basis:
#   • FinRL (Liu et al., NeurIPS 2020) — ensemble of A2C + PPO + DDPG/TD3
#   • "Benchmarking RL for Portfolio Optimization" (2024) — A2C top cumulative
#     reward; TD3 most balanced/diversified; SAC natural entropy bonus
#   • TD3 with Sortino reward + 0.1% transaction cost (ScienceDirect 2024)
#   • Volatility regime detection as observation feature
#   • 45% single-asset hard cap
# ============================================================================

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, TD3, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


MAX_SINGLE_WEIGHT  = 0.45
TRANSACTION_COST   = 0.001   # 0.1% per trade — standard in FinRL benchmarks
RISK_FREE_RATE_ANN = 0.05    # annualised, used in Sortino calculation


def _cap_and_renormalize(weights, cap=MAX_SINGLE_WEIGHT):
    """Iterative cap + renormalise. Guaranteed to converge in ≤ n steps."""
    w = weights.copy()
    for _ in range(len(w)):
        excess = np.maximum(w - cap, 0.0)
        if excess.sum() < 1e-9:
            break
        w    = np.minimum(w, cap)
        pool = excess.sum()
        free = w < cap
        if free.sum() == 0:
            w[:] = 1.0 / len(w)
            break
        w[free] += pool * (w[free] / w[free].sum())
    total = w.sum()
    return w / total if total > 0 else w


def _sortino_ratio(returns_arr, rf_daily=RISK_FREE_RATE_ANN / 252):
    """
    Sortino ratio — penalises ONLY downside volatility.
    More appropriate than Sharpe for skewed financial return distributions.
    (Research: better reward signal for portfolio RL agents)
    """
    if len(returns_arr) < 5:
        return 0.0
    excess     = returns_arr - rf_daily
    downside   = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2)) + 1e-8
    return float(np.mean(excess) / downside_std * np.sqrt(252))


def _volatility_regime(returns_arr, window=20):
    """
    0 = low vol regime  (rolling std < 1st quartile of history)
    1 = normal regime
    2 = high vol regime (rolling std > 3rd quartile of history)
    Returns normalised value in [0, 1].
    """
    if len(returns_arr) < window:
        return 0.5
    recent_vol = np.std(returns_arr[-window:])
    hist_vol   = np.std(returns_arr)
    if hist_vol < 1e-10:
        return 0.5
    return float(np.clip(recent_vol / (2.0 * hist_vol), 0.0, 1.0))


class ProgressCallback(BaseCallback):
    """
    Writes progress JSON to disk every 100 steps.
    offset_pct / scale_pct allow multiple agents to share one progress bar.
    """
    def __init__(self, total_timesteps, progress_file, offset_pct=0.0, scale_pct=1.0):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_file   = progress_file
        self.offset_pct      = offset_pct
        self.scale_pct       = scale_pct
        self.current_reward  = 0.0

    def _on_step(self):
        self.current_reward += self.locals.get("rewards", [0])[0]
        raw = min(self.num_timesteps / self.total_timesteps, 1.0)
        overall = self.offset_pct + raw * self.scale_pct
        if self.num_timesteps % 100 == 0:
            with open(self.progress_file, "w") as f:
                json.dump({
                    "progress":        round(overall * 100, 1),
                    "timestep":        self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "reward":          round(float(self.current_reward), 4),
                    "status":          "training",
                }, f)
        return True


class PortfolioEnv(gym.Env):
    """
    Industry-grade portfolio environment.

    Observation (per step, normalised)
    -----------------------------------
    For each asset:
      • 30-day daily returns window
      • 5-day momentum (short trend)
      • 20-day momentum (medium trend)
      • Rolling RSI [0,1]
      • Rolling MACD histogram [-1,1]
      • Bollinger Band %B [0,1]
      • Fundamental score [0,1]   ← pre-fetched once
      • News sentiment [-1,1]     ← Groq-scored headlines
    Global:
      • Volatility regime [0,1]   ← rolling std vs history
      • Current weights (n)
      • Normalised portfolio value (1)

    Reward (research-backed composite)
    ------------------------------------
    Base:   daily portfolio return − transaction costs (0.1% per rebalance)
    +0.02 × Sortino ratio (penalises only downside vol, not upside)
    −sector_risk_avg × 0.015 × max_drawdown
    +0.008 × weight entropy (diversification)
    −0.005 × concentration² (soft pressure below 45% cap)
    +0.003 × fundamental alignment
    +0.003 × sentiment alignment

    Constraints
    -----------
    Hard 45% cap via _cap_and_renormalize on every step.
    """

    WINDOW          = 30
    INITIAL_CAPITAL = 100_000.0

    def __init__(self, price_data, ticker_features):
        super().__init__()
        self.price_data  = price_data.copy()
        self.returns     = price_data.pct_change().fillna(0)
        self.prices_arr  = price_data.values
        self.n_assets    = len(price_data.columns)
        self.tickers     = list(price_data.columns)
        self.ticker_features = ticker_features

        self._mom5  = price_data.pct_change(5).fillna(0).values
        self._mom20 = price_data.pct_change(20).fillna(0).values

        self._fund_scores  = np.array([ticker_features.get(t, {}).get("fundamental_score", 0.5) for t in self.tickers], dtype=np.float32)
        self._sentiments   = np.array([ticker_features.get(t, {}).get("news_sentiment",    0.0) for t in self.tickers], dtype=np.float32)
        self._sector_risks = np.array([ticker_features.get(t, {}).get("sector_risk",       1.0) for t in self.tickers], dtype=np.float32)

        # obs: 30n + 5n_dynamic + 2n_static + n_weights + 1_value + 1_regime
        obs_size = self.WINDOW * self.n_assets + 7 * self.n_assets + self.n_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space      = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step      = self.WINDOW
        self.portfolio_value   = self.INITIAL_CAPITAL
        self.peak_value        = self.INITIAL_CAPITAL
        self.weights           = np.ones(self.n_assets) / self.n_assets
        self.prev_weights      = self.weights.copy()
        self.portfolio_history = [self.INITIAL_CAPITAL]
        return self._get_obs(), {}

    def _get_obs(self):
        i      = self.current_step
        recent = self.returns.iloc[i - self.WINDOW: i].values
        if recent.shape[0] < self.WINDOW:
            pad    = np.zeros((self.WINDOW - recent.shape[0], self.n_assets))
            recent = np.vstack([pad, recent])
        mom5  = self._mom5[i]  if i < len(self._mom5)  else np.zeros(self.n_assets)
        mom20 = self._mom20[i] if i < len(self._mom20) else np.zeros(self.n_assets)

        lb = min(i, 60)
        rsi_v  = np.zeros(self.n_assets, dtype=np.float32)
        macd_v = np.zeros(self.n_assets, dtype=np.float32)
        bb_v   = np.zeros(self.n_assets, dtype=np.float32)
        for j in range(self.n_assets):
            sl = self.prices_arr[max(0, i - lb): i + 1, j]
            rsi_v[j]  = _rolling_rsi(sl)
            macd_v[j] = _rolling_macd(sl)
            bb_v[j]   = _bollinger_pct(sl)

        # Volatility regime (global market feature)
        all_rets = self.returns.values[:i].mean(axis=1)   # equal-weight market return
        regime   = _volatility_regime(all_rets)

        return np.concatenate([
            recent.flatten(), mom5, mom20,
            rsi_v, macd_v, bb_v,
            self._fund_scores, self._sentiments,
            self.weights,
            [self.portfolio_value / self.INITIAL_CAPITAL, regime],
        ]).astype(np.float32)

    def step(self, action):
        # Normalise → cap
        action = np.clip(action, 0, None)
        s      = action.sum()
        raw_w  = action / s if s > 0 else np.ones(self.n_assets) / self.n_assets
        self.weights = _cap_and_renormalize(raw_w, MAX_SINGLE_WEIGHT)

        if self.current_step >= len(self.returns):
            return self._get_obs(), 0.0, True, False, {}

        # Transaction cost (0.1% × total weight turnover)
        turnover    = float(np.sum(np.abs(self.weights - self.prev_weights)))
        tc_cost     = TRANSACTION_COST * turnover
        self.prev_weights = self.weights.copy()

        # Daily P&L
        daily_rets       = self.returns.iloc[self.current_step].values
        portfolio_return = float(np.dot(self.weights, daily_rets)) - tc_cost
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)

        # Drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        # Composite reward
        reward = portfolio_return  # base (includes transaction cost)

        if len(self.portfolio_history) >= 10:
            hist_arr = np.array(self.portfolio_history[-60:])
            rets_arr = np.diff(hist_arr) / (hist_arr[:-1] + 1e-8)
            # Sortino (research: better than Sharpe for skewed fin returns)
            sortino  = _sortino_ratio(rets_arr)
            reward  += 0.02 * sortino

        # Sector-risk-adjusted drawdown penalty
        avg_sector_risk = float(np.dot(self.weights, self._sector_risks))
        reward -= avg_sector_risk * 0.015 * drawdown

        # Entropy bonus (diversification)
        entropy = -float(np.sum(self.weights * np.log(self.weights + 1e-8)))
        reward += 0.008 * entropy

        # Concentration penalty (soft pressure below hard cap)
        concentration = float(np.sum(np.maximum(self.weights - 0.30, 0.0) ** 2))
        reward -= 0.005 * concentration

        # Fundamental alignment bonus
        reward += 0.003 * float(np.dot(self.weights, self._fund_scores - 0.5))

        # Sentiment alignment bonus
        reward += 0.003 * float(np.dot(self.weights, self._sentiments))

        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Ensemble Agent Factory
# ---------------------------------------------------------------------------

def _make_a2c(env, timesteps):
    """
    A2C — best cumulative rewards in 2024 FinRL benchmarks.
    On-policy, fast, good for non-stationary markets.
    """
    return A2C(
        "MlpPolicy", env,
        verbose       = 0,
        learning_rate = 7e-4,
        n_steps       = 5,
        gamma         = 0.99,
        gae_lambda    = 1.0,
        ent_coef      = 0.01,
        policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    )


def _make_sac(env, n_assets):
    """
    SAC — built-in entropy maximisation promotes diversification naturally.
    Off-policy, sample efficient, great for continuous portfolio weights.
    """
    return SAC(
        "MlpPolicy", env,
        verbose       = 0,
        learning_rate = 3e-4,
        buffer_size   = 100_000,
        batch_size    = 256,
        tau           = 0.005,
        gamma         = 0.99,
        ent_coef      = "auto",   # auto-tune entropy temperature
        policy_kwargs = dict(net_arch=[256, 256]),
    )


def _make_td3(env, n_assets):
    """
    TD3 — most balanced/diversified holdings per 2024 benchmarks.
    Twin critics eliminate overestimation bias. Best for risk-sensitive tasks.
    Includes action noise for better exploration.
    """
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_assets),
        sigma = 0.1 * np.ones(n_assets),
    )
    return TD3(
        "MlpPolicy", env,
        verbose       = 0,
        learning_rate = 1e-3,
        buffer_size   = 100_000,
        batch_size    = 256,
        tau           = 0.005,
        gamma         = 0.99,
        action_noise  = action_noise,
        policy_kwargs = dict(net_arch=[256, 256]),
    )


class RLPortfolioAgent:
    """
    FinRL-style Ensemble Agent.

    Trains three independent agents:
      1. A2C  — on-policy, highest cumulative returns
      2. SAC  — off-policy, entropy bonus for diversification
      3. TD3  — off-policy, twin critics, most balanced holdings

    Final allocation = weighted average of all three predictions:
      weights = 0.35×A2C + 0.30×SAC + 0.35×TD3

    Research: Liu et al. NeurIPS 2020 show ensemble strategy outperforms
    any single agent and the market index on Dow Jones 30.
    """

    AGENT_WEIGHTS = {"A2C": 0.35, "SAC": 0.30, "TD3": 0.35}

    MODEL_DIR    = os.path.join(os.path.dirname(__file__), "rl_models")
    PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "rl_progress")

    def __init__(self):
        os.makedirs(self.MODEL_DIR,    exist_ok=True)
        os.makedirs(self.PROGRESS_DIR, exist_ok=True)

    def _fetch_data(self, tickers, period="2y"):
        frames = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period=period)
                if not hist.empty:
                    hist.index = pd.to_datetime(hist.index.date)
                    frames[t]  = hist["Close"]
            except Exception:
                continue
        if len(frames) < 2:
            raise ValueError("Need at least 2 tickers with valid data.")
        df = pd.DataFrame(frames).ffill().dropna()
        if len(df) < 60:
            raise ValueError("Not enough aligned data. Check tickers.")
        return df

    def _write_progress(self, progress_file, progress, status, timestep=0,
                        total_timesteps=0, reward=0, extra=None):
        data = {
            "progress": round(progress, 1), "status": status,
            "timestep": timestep, "total_timesteps": total_timesteps,
            "reward":   round(float(reward), 4),
        }
        if extra:
            data.update(extra)
        with open(progress_file, "w") as f:
            json.dump(data, f)

    def train(self, tickers, timesteps=10000, session_id="default"):
        if not GYM_AVAILABLE or not SB3_AVAILABLE:
            raise ImportError("Install: pip install stable-baselines3 gymnasium")

        progress_file = os.path.join(self.PROGRESS_DIR, f"{session_id}.json")
        self._write_progress(progress_file, 0, "fetching_data", total_timesteps=timesteps)

        price_data = self._fetch_data(tickers)

        self._write_progress(progress_file, 0, "fetching_fundamentals", total_timesteps=timesteps)
        ticker_features = fetch_ticker_features(tickers)

        # Each of 3 agents gets 1/3 of the progress bar
        agent_share = 1.0 / 3.0

        trained_models = {}
        all_rewards    = {}

        for idx, (agent_name, offset) in enumerate([("A2C", 0.0), ("SAC", agent_share), ("TD3", 2 * agent_share)]):
            self._write_progress(
                progress_file,
                offset * 100,
                f"training_{agent_name}",
                total_timesteps=timesteps,
                extra={"current_agent": agent_name, "agent_index": idx + 1}
            )

            env      = DummyVecEnv([lambda: PortfolioEnv(price_data, ticker_features)])
            callback = ProgressCallback(timesteps, progress_file, offset_pct=offset, scale_pct=agent_share)

            if agent_name == "A2C":
                model = _make_a2c(env, timesteps)
            elif agent_name == "SAC":
                # SAC needs non-vectorised env for some versions
                single_env = PortfolioEnv(price_data, ticker_features)
                model = _make_sac(single_env, len(tickers))
            else:  # TD3
                single_env = PortfolioEnv(price_data, ticker_features)
                model = _make_td3(single_env, len(tickers))

            model.learn(total_timesteps=timesteps, callback=callback)
            model.save(os.path.join(self.MODEL_DIR, f"{session_id}_{agent_name}"))

            trained_models[agent_name] = model
            all_rewards[agent_name]    = float(callback.current_reward)

        self._write_progress(progress_file, 99, "evaluating", total_timesteps=timesteps)

        # Evaluate each agent and collect final weights
        agent_allocations    = {}
        agent_pf_histories   = {}
        agent_portfolio_dates = {}

        for agent_name, model in trained_models.items():
            alloc, pf_hist, pf_dates = self._evaluate_single(model, price_data, ticker_features)
            agent_allocations[agent_name]    = alloc
            agent_pf_histories[agent_name]   = pf_hist
            agent_portfolio_dates[agent_name] = pf_dates

        # Ensemble: weighted average of allocations
        ensemble_weights = np.zeros(len(tickers))
        for agent_name, w in self.AGENT_WEIGHTS.items():
            if agent_name in agent_allocations:
                alloc_arr = np.array([agent_allocations[agent_name].get(t, 0.0) for t in tickers])
                ensemble_weights += w * alloc_arr
        # Re-normalise and re-cap
        ensemble_weights = _cap_and_renormalize(ensemble_weights / (ensemble_weights.sum() + 1e-8))
        ensemble_alloc   = {t: round(float(w), 4) for t, w in zip(tickers, ensemble_weights)}

        # Use A2C portfolio history as primary (best cumulative returns)
        primary_agent    = "A2C"
        portfolio_history = agent_pf_histories[primary_agent]
        portfolio_dates   = agent_portfolio_dates[primary_agent]
        final_value       = portfolio_history[-1] if portfolio_history else 100000
        total_return      = ((final_value / 100000) - 1) * 100

        # Per-agent performance metrics
        agent_metrics = {}
        for agent_name, pf_hist in agent_pf_histories.items():
            if len(pf_hist) > 1:
                rets = np.diff(pf_hist) / (np.array(pf_hist[:-1]) + 1e-8)
                agent_metrics[agent_name] = {
                    "final_value":  round(pf_hist[-1], 2),
                    "total_return": round((pf_hist[-1] / 100000 - 1) * 100, 2),
                    "sharpe":       round(float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)), 3),
                    "sortino":      round(_sortino_ratio(rets), 3),
                    "max_drawdown": round(float(self._max_drawdown(pf_hist)), 4),
                    "allocation":   agent_allocations[agent_name],
                }

        result = {
            "progress": 100, "status": "complete",
            "timestep": timesteps, "total_timesteps": timesteps,
            "reward":   float(np.mean(list(all_rewards.values()))),
            "allocation":        ensemble_alloc,
            "portfolio_history": portfolio_history,
            "portfolio_dates":   portfolio_dates,
            "tickers":           tickers,
            "ticker_features":   {
                t: {k: round(v, 3) for k, v in feats.items()}
                for t, feats in ticker_features.items()
            },
            "agent_metrics":     agent_metrics,
            "final_value":       final_value,
            "total_return":      total_return,
        }
        with open(progress_file, "w") as f:
            json.dump(result, f)
        return result

    def _evaluate_single(self, model, price_data, ticker_features):
        env    = PortfolioEnv(price_data, ticker_features)
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _          = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        last_weights      = env.weights.copy()
        allocation        = {t: round(float(w), 4) for t, w in zip(price_data.columns, last_weights)}
        portfolio_history = [round(v, 2) for v in env.portfolio_history]

        start_idx       = env.WINDOW
        end_idx         = start_idx + len(portfolio_history)
        date_index      = price_data.index[start_idx:end_idx]
        portfolio_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in date_index]
        while len(portfolio_dates) < len(portfolio_history):
            portfolio_dates.append("")

        return allocation, portfolio_history, portfolio_dates

    def _max_drawdown(self, pf_history):
        arr  = np.array(pf_history)
        peak = np.maximum.accumulate(arr)
        dd   = (peak - arr) / (peak + 1e-8)
        return float(dd.max())

    def get_progress(self, session_id):
        progress_file = os.path.join(self.PROGRESS_DIR, f"{session_id}.json")
        if not os.path.exists(progress_file):
            return {"progress": 0, "status": "not_started"}
        with open(progress_file) as f:
            return json.load(f)


_rl_agent = RLPortfolioAgent()
