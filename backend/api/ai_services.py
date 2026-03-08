import os
import json
import numpy as np
import pandas as pd
from groq import Groq
import yfinance as yf

# --- PyTorch (replaces tensorflow) ---
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --- fastembed (replaces sentence-transformers — no 400MB download, same quality) ---
try:
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception:
    embedding_model = None

# --- PyPortfolioOpt (replaces random mock optimizer) ---
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ---------------------------------------------------------------------------
# LSTM Model Definition (PyTorch — replaces keras Sequential)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Groq LLM Service (unchanged — llama-3.3-70b-versatile is still latest)
# ---------------------------------------------------------------------------

api_key     = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None


class GroqService:
    @staticmethod
    def chat(message):
        if not groq_client:
            return "Groq API Key not found. This is a mock response."
        try:
            if isinstance(message, list):
                messages = message
            else:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful financial assistant for the QuantVision app. "
                            "Use the provided context to answer questions."
                        ),
                    },
                    {"role": "user", "content": message},
                ]

            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error interacting with AI: {str(e)}"

    @staticmethod
    def analyze_allocation(tickers, allocation, risk_tolerance):
        """Generates AI commentary on the optimized portfolio allocation."""
        if not groq_client:
            return "Groq API Key not found. This is a mock response."
        try:
            prompt = f"""
            You are a portfolio manager. Analyze the following optimized allocation:

            Risk Tolerance: {risk_tolerance} (0=Conservative, 1=Aggressive)
            Allocation:
            {allocation}

            Explain WHY this allocation makes sense given the risk tolerance.
            Highlight the role of key assets (e.g. why is there so much in Tech? or Gold?).
            Keep it concise (3-4 sentences).
            """
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a financial expert."},
                    {"role": "user",   "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Could not generate analysis: {str(e)}"


# ---------------------------------------------------------------------------
# Trend Predictor  (yfinance news API fixed for >= 0.2.40)
# ---------------------------------------------------------------------------

class TrendPredictor:
    @staticmethod
    def get_peers(symbol, sector=None):
        is_indian = ".NS" in symbol or ".BO" in symbol

        if is_indian:
            peers = ["NIFTYBEES.NS", "GOLDBEES.NS"]
            if sector:
                sec = sector.lower()
                if "energy" in sec or "oil" in sec:
                    peers = ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"]
                elif "technology" in sec or "computers" in sec:
                    peers = ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"]
                elif "financial" in sec or "bank" in sec:
                    peers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"]
                elif "auto" in sec:
                    peers = ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS"]
                elif "healthcare" in sec or "pharma" in sec:
                    peers = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"]
                elif "consumer" in sec:
                    peers = ["HINDUNILVR.NS", "ITC.NS", "TITAN.NS"]
            return [p for p in peers if p != symbol]

        if "AAPL" in symbol: return ["MSFT", "GOOG", "AMZN"]
        if "TSLA" in symbol: return ["F", "GM", "RIVN"]
        return ["SPY", "QQQ"]

    @staticmethod
    def _parse_news(raw_news):
        """
        Parse yfinance news — handles both old dict format and new
        nested 'content' format introduced in yfinance >= 0.2.40.
        """
        parsed = []
        for n in raw_news:
            # New format: everything is nested under n['content']
            content = n.get("content", n)
            parsed.append({
                "title":     content.get("title",    n.get("title",     "No Title")),
                "publisher": (
                    content.get("provider", {}).get("displayName")
                    or n.get("publisher", "Unknown")
                ),
                "link": (
                    content.get("canonicalUrl", {}).get("url")
                    or n.get("link", "#")
                ),
            })
        return parsed

    @staticmethod
    def predict(symbol):
        try:
            ticker     = yf.Ticker(symbol)
            hist_daily = ticker.history(period="6mo")
            hist_weekly = ticker.history(period="1y", interval="1wk")[-30:]
            info       = ticker.info

            if hist_daily.empty:
                return {"error": f"No data found for {symbol}. Try adding .NS for Indian stocks."}

            # ---- Financials ----
            def extract_financials(info_dict):
                if not info_dict:
                    return {}
                return {
                    "Market Cap":      info_dict.get("marketCap",        "N/A"),
                    "Current Price":   info_dict.get("currentPrice",     info_dict.get("regularMarketPrice", "N/A")),
                    "EPS (Trailing)":  info_dict.get("trailingEps",      "N/A"),
                    "PE Ratio":        info_dict.get("trailingPE",       "N/A"),
                    "Debt to Equity":  info_dict.get("debtToEquity",     "N/A"),
                    "Total Debt":      info_dict.get("totalDebt",        "N/A"),
                    "Revenue Growth":  info_dict.get("revenueGrowth",    "N/A"),
                    "Sector":          info_dict.get("sector",           "N/A"),
                }

            financials = extract_financials(info)
            sector     = info.get("sector", "")

            # ---- News (fixed for yfinance >= 0.2.40) ----
            raw_news = ticker.news[:3] if ticker.news else []
            news     = TrendPredictor._parse_news(raw_news)
            news_text = "\n".join([f"- {n['title']} ({n['publisher']})" for n in news])

            # ---- Peers ----
            peers        = TrendPredictor.get_peers(symbol, sector)
            peer_data    = []
            peer_history = {}

            target_data           = financials.copy()
            target_data["Symbol"] = symbol
            peer_data.append(target_data)
            peer_history[symbol]  = hist_daily["Close"].tolist()

            for p in peers:
                try:
                    p_ticker = yf.Ticker(p)
                    p_info   = p_ticker.info
                    p_fin    = extract_financials(p_info)
                    p_fin["Symbol"] = p
                    peer_data.append(p_fin)
                    p_hist = p_ticker.history(period="6mo")
                    if not p_hist.empty:
                        peer_history[p] = p_hist["Close"].tolist()
                except Exception:
                    continue

            # ---- Technical Indicators ----
            df    = hist_daily.copy()
            close = df["Close"]

            if len(close) < 50:
                return {"error": "Not enough historical data for technical analysis."}

            # RSI
            delta = close.diff()
            gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs    = gain / loss
            rsi   = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

            # MACD
            exp1          = close.ewm(span=12, adjust=False).mean()
            exp2          = close.ewm(span=26, adjust=False).mean()
            macd          = exp1 - exp2
            signal_line   = macd.ewm(span=9, adjust=False).mean()
            current_macd  = float(macd.iloc[-1])
            current_signal = float(signal_line.iloc[-1])

            sma_50       = float(close.mean())
            recent_close = float(close.iloc[-1])

            signals = []
            if current_rsi < 30:              signals.append("Oversold (Bullish)")
            if current_rsi > 70:              signals.append("Overbought (Bearish)")
            if current_macd > current_signal: signals.append("MACD Bullish Crossover")
            if current_macd < current_signal: signals.append("MACD Bearish Crossover")
            if recent_close > sma_50:         signals.append("Price > 30d Avg")

            bullish_votes = sum(1 for s in signals if "Bullish" in s or ">" in s)
            bearish_votes = sum(1 for s in signals if "Bearish" in s or "<" in s)

            if bullish_votes > bearish_votes:
                trend        = "UP"
                change_pct   = np.random.uniform(0.01, 0.05)
            elif bearish_votes > bullish_votes:
                trend        = "DOWN"
                change_pct   = np.random.uniform(-0.05, -0.01)
            else:
                trend        = "NEUTRAL"
                change_pct   = np.random.uniform(-0.005, 0.005)

            predicted_price      = recent_close * (1 + change_pct)
            last_30_days_prices  = hist_daily["Close"].tail(30).tolist()

            # ---- AI Reasoning ----
            prompt = f"""
            Analyze the stock {symbol} (Sector: {sector}).

            TECHNICAL DATA:
            - Current Price: {recent_close}
            - RSI (14): {current_rsi:.2f}
            - MACD Line: {current_macd:.4f}
            - Signal Line: {current_signal:.4f}
            - Observed Signals: {', '.join(signals)}
            - Calculated Trend: {trend}

            FUNDAMENTAL DATA:
            - Financials: {financials}

            CONTEXT:
            - Recent News: {news_text}
            - Peer Data: {peer_data}
            - Last 30 Days Prices: {last_30_days_prices}

            TASK:
            Provide a comprehensive market analysis.
            CRITICAL: Your reasoning MUST align with the Calculated Trend ({trend}) and the Technical Signals ({signals}).
            If the Trend is UP, find bullish arguments in the technicals and news.
            If the Trend is DOWN, highlight the bearish risks.
            """
            ai_reasoning = GroqService.chat(prompt)

            return {
                "symbol":         symbol,
                "current_price":  float(recent_close),
                "predicted_price": float(predicted_price),
                "trend":          trend,
                "financials":     financials,
                "news":           news,
                "peer_financials": peer_data,
                "peer_history":   peer_history,
                "weekly_data":    hist_weekly["Close"].tolist(),
                "technicals": {
                    "rsi":     float(current_rsi),
                    "macd":    float(current_macd),
                    "signal":  float(current_signal),
                    "signals": signals,
                },
                "ohlc_data": json.loads(
                    hist_daily.to_json(orient="split", date_format="iso")
                ),
                "reasoning": ai_reasoning,
            }

        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Portfolio Optimizer  (replaces random mock — now uses real Markowitz)
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    @staticmethod
    def optimize(tickers, risk_tolerance):
        """
        Real mean-variance optimisation using PyPortfolioOpt.
        risk_tolerance 0.0–0.39 → minimum volatility portfolio
        risk_tolerance 0.4–1.0  → maximum Sharpe ratio portfolio
        Falls back to equal-weight if data is insufficient.
        """
        try:
            # Fetch 2 years of adjusted close prices
            raw = yf.download(tickers, period="2y", auto_adjust=True, progress=False)

            # yfinance returns MultiIndex columns when len(tickers) > 1
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw["Close"]
            else:
                raw = raw[["Close"]] if "Close" in raw.columns else raw

            raw = raw.dropna()

            if raw.empty or len(raw) < 60:
                raise ValueError("Not enough historical data for optimisation.")

            # Ensure columns match requested tickers (yfinance may reorder)
            available = [t for t in tickers if t in raw.columns]
            if len(available) < 2:
                raise ValueError("Need at least 2 tickers with data.")
            raw = raw[available]

            mu = expected_returns.mean_historical_return(raw)
            S  = risk_models.sample_cov(raw)

            ef = EfficientFrontier(mu, S)

            if risk_tolerance < 0.4:
                ef.min_volatility()
            else:
                ef.max_sharpe(risk_free_rate=0.05)

            weights = ef.clean_weights()
            # Return only non-trivial weights
            return {k: round(v, 4) for k, v in weights.items() if v > 0.001}

        except Exception:
            # Graceful fallback: equal weight across all tickers
            n = len(tickers)
            return {t: round(1.0 / n, 4) for t in tickers}
