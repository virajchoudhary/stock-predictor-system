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

# --- fastembed (replaces sentence-transformers) ---
try:
    from fastembed import TextEmbedding
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception:
    embedding_model = None

# --- Riskfolio-Lib (replaces PyPortfolioOpt — HRP, CVaR, Mean-Variance) ---
import riskfolio as rp

# --- ta (replaces manual RSI/MACD — 130+ indicators, one line each) ---
import ta

# --- Polygon.io (official market data API — replaces yfinance for price data) ---
from polygon import RESTClient
polygon_api_key = os.environ.get("POLYGON_API_KEY")
polygon_client  = RESTClient(api_key=polygon_api_key) if polygon_api_key else None


# ---------------------------------------------------------------------------
# LSTM Model Definition (PyTorch)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ---------------------------------------------------------------------------
# Groq LLM Service
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
# Market Data — Polygon.io with yfinance fallback
# ---------------------------------------------------------------------------

def get_price_history(symbol, period="6mo"):
    """
    Fetch OHLCV price history.
    Primary:  Polygon.io (official API, reliable, no scraping)
    Fallback: yfinance
    """
    if polygon_client:
        try:
            from datetime import datetime, timedelta
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
            days        = period_days.get(period, 180)
            end_date    = datetime.now()
            start_date  = end_date - timedelta(days=days)

            aggs = polygon_client.get_aggs(
                ticker     = symbol.upper(),
                multiplier = 1,
                timespan   = "day",
                from_      = start_date.strftime("%Y-%m-%d"),
                to         = end_date.strftime("%Y-%m-%d"),
                limit      = 50000,
            )

            if aggs:
                df = pd.DataFrame([{
                    "Open":   a.open,
                    "High":   a.high,
                    "Low":    a.low,
                    "Close":  a.close,
                    "Volume": a.volume,
                } for a in aggs],
                index=pd.to_datetime([a.timestamp for a in aggs], unit="ms"))
                df.index.name = "Date"
                if not df.empty:
                    return df
        except Exception:
            pass  # Fall through to yfinance

    # yfinance fallback
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period)


def get_ticker_info(symbol):
    """
    Fetch company info.
    Primary:  Polygon.io
    Fallback: yfinance
    """
    if polygon_client:
        try:
            detail = polygon_client.get_ticker_details(symbol.upper())
            if detail:
                return {
                    "marketCap":     getattr(detail, "market_cap",       "N/A"),
                    "currentPrice":  "N/A",
                    "trailingEps":   "N/A",
                    "trailingPE":    "N/A",
                    "debtToEquity":  "N/A",
                    "totalDebt":     "N/A",
                    "revenueGrowth": "N/A",
                    "sector":        getattr(detail, "sic_description",  "N/A"),
                    "name":          getattr(detail, "name",             symbol),
                }
        except Exception:
            pass

    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Trend Predictor — ta library for indicators, Polygon for data
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
        """Handles both old and new yfinance >= 0.2.40 news format."""
        parsed = []
        for n in raw_news:
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
    def _lstm_predict(symbol, hidden_size=64, num_layers=2, learning_rate=0.001, epochs=30, dropout=0.0, seq_len=20):
        """
        Trains an LSTM on recent data using given hyperparams and returns
        the next-day predicted Close price (unscaled).
        Returns None on any failure.
        """
        try:
            from .evolution import get_train_val_data
            result = get_train_val_data(symbol, seq_len=seq_len)
            X_train, y_train, X_val, y_val, scaler, input_size = result

            model     = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                out  = model(X_train)
                loss = criterion(out, y_train)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                # Use the last validation sequence as input for next-day prediction
                last_seq    = X_val[-1].unsqueeze(0)  # shape: (1, seq_len, input_size)
                pred_scaled = model(last_seq).item()

            # Inverse transform: only Close (column 0) was predicted
            dummy       = np.zeros((1, input_size))
            dummy[0, 0] = pred_scaled
            pred_price  = scaler.inverse_transform(dummy)[0, 0]
            return float(pred_price)
        except Exception:
            return None

    @staticmethod
    def predict(symbol, hidden_size=64, num_layers=2, learning_rate=0.001, epochs=30, dropout=0.0, seq_len=20, hyperparams_source='default'):
        try:
            # ---- Price History ----
            hist_daily  = get_price_history(symbol, period="6mo")
            hist_weekly = get_price_history(symbol, period="1y")
            if hist_weekly is not None and not hist_weekly.empty:
                hist_weekly = hist_weekly.resample("W").last().tail(30)

            if hist_daily is None or hist_daily.empty:
                return {"error": f"No data found for {symbol}."}

            # ---- Company Info ----
            info    = get_ticker_info(symbol)
            sector  = info.get("sector", info.get("sic_description", ""))

            def extract_financials(info_dict):
                if not info_dict:
                    return {}
                return {
                    "Market Cap":     info_dict.get("marketCap",          "N/A"),
                    "Current Price":  info_dict.get("currentPrice",
                                      info_dict.get("regularMarketPrice", "N/A")),
                    "EPS (Trailing)": info_dict.get("trailingEps",        "N/A"),
                    "PE Ratio":       info_dict.get("trailingPE",         "N/A"),
                    "Debt to Equity": info_dict.get("debtToEquity",       "N/A"),
                    "Total Debt":     info_dict.get("totalDebt",          "N/A"),
                    "Revenue Growth": info_dict.get("revenueGrowth",      "N/A"),
                    "Sector":         info_dict.get("sector",
                                      info_dict.get("sic_description",   "N/A")),
                }

            financials = extract_financials(info)

            # ---- News ----
            try:
                ticker   = yf.Ticker(symbol)
                raw_news = ticker.news[:3] if ticker.news else []
                news     = TrendPredictor._parse_news(raw_news)
            except Exception:
                news = []
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
                    p_hist = get_price_history(p, period="6mo")
                    p_info = get_ticker_info(p)
                    p_fin  = extract_financials(p_info)
                    p_fin["Symbol"] = p
                    peer_data.append(p_fin)
                    if p_hist is not None and not p_hist.empty:
                        peer_history[p] = p_hist["Close"].tolist()
                except Exception:
                    continue

            # ---- Technical Indicators (ta library — replaces manual calculations) ----
            if len(hist_daily) < 50:
                return {"error": "Not enough historical data for technical analysis."}

            close  = hist_daily["Close"]
            high   = hist_daily["High"]
            low    = hist_daily["Low"]
            volume = hist_daily["Volume"]

            # RSI
            current_rsi = float(
                ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
            ) if not pd.isna(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]) else 50.0

            # MACD
            macd_obj       = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            current_macd   = float(macd_obj.macd().iloc[-1])
            current_signal = float(macd_obj.macd_signal().iloc[-1])

            # Bollinger Bands
            bb        = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_upper  = float(bb.bollinger_hband().iloc[-1])
            bb_lower  = float(bb.bollinger_lband().iloc[-1])
            bb_mid    = float(bb.bollinger_mavg().iloc[-1])

            # ATR
            current_atr = float(
                ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
                .average_true_range().iloc[-1]
            )

            # SMA 50 & SMA 200
            sma_50  = float(ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])
            sma_200_val = ta.trend.SMAIndicator(close=close, window=200).sma_indicator().iloc[-1]
            sma_200 = float(sma_200_val) if not pd.isna(sma_200_val) else None

            recent_close = float(close.iloc[-1])

            # ---- Signals ----
            signals = []
            if current_rsi < 30:                          signals.append("Oversold (Bullish)")
            if current_rsi > 70:                          signals.append("Overbought (Bearish)")
            if current_macd > current_signal:             signals.append("MACD Bullish Crossover")
            if current_macd < current_signal:             signals.append("MACD Bearish Crossover")
            if recent_close > sma_50:                     signals.append("Price > SMA50 (Bullish)")
            if sma_200 and recent_close > sma_200:        signals.append("Price > SMA200 (Bullish)")
            if recent_close > bb_upper:                   signals.append("Above Bollinger Upper (Overbought)")
            if recent_close < bb_lower:                   signals.append("Below Bollinger Lower (Oversold)")

            bullish_votes = sum(1 for s in signals if "Bullish" in s or "Oversold" in s)
            bearish_votes = sum(1 for s in signals if "Bearish" in s or "Overbought" in s)

            # LSTM price prediction — uses evolved hyperparams if available
            lstm_price = TrendPredictor._lstm_predict(
                symbol,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
                epochs=epochs,
                dropout=dropout,
                seq_len=seq_len
            )

            if lstm_price is not None:
                if lstm_price > recent_close:
                    trend = "UP"
                elif lstm_price < recent_close:
                    trend = "DOWN"
                else:
                    trend = "NEUTRAL"
            else:
                if bullish_votes > bearish_votes:
                    trend = "UP"
                elif bearish_votes > bullish_votes:
                    trend = "DOWN"
                else:
                    trend = "NEUTRAL"

            predicted_price = lstm_price if lstm_price is not None else recent_close * 1.001
            last_30_days_prices = hist_daily["Close"].tail(30).tolist()

            # ---- AI Reasoning ----
            prompt = f"""
            Analyze the stock {symbol} (Sector: {sector}).

            TECHNICAL DATA:
            - Current Price:   {recent_close}
            - RSI (14):        {current_rsi:.2f}
            - MACD:            {current_macd:.4f}
            - Signal Line:     {current_signal:.4f}
            - SMA 50:          {sma_50:.2f}
            - SMA 200:         {f'{sma_200:.2f}' if sma_200 else 'N/A'}
            - Bollinger Upper: {bb_upper:.2f}
            - Bollinger Lower: {bb_lower:.2f}
            - ATR (14):        {current_atr:.2f}
            - Signals:         {', '.join(signals)}
            - Trend:           {trend}

            FUNDAMENTAL DATA: {financials}
            RECENT NEWS: {news_text}
            PEER DATA: {peer_data}

            Provide a comprehensive analysis aligned with the trend ({trend}).
            Reference Bollinger Bands and ATR in your volatility assessment.
            """
            ai_reasoning = GroqService.chat(prompt)

            return {
                "symbol":          symbol,
                "current_price":   float(recent_close),
                "predicted_price": float(predicted_price),
                "trend":           trend,
                "financials":      financials,
                "news":            news,
                "peer_financials": peer_data,
                "peer_history":    peer_history,
                "weekly_data":     hist_weekly["Close"].tolist() if hist_weekly is not None and not hist_weekly.empty else [],
                "technicals": {
                    "rsi":     float(current_rsi),
                    "macd":    float(current_macd),
                    "signal":  float(current_signal),
                    "sma_50":  float(sma_50),
                    "sma_200": float(sma_200) if sma_200 else None,
                    "bb_upper": float(bb_upper),
                    "bb_lower": float(bb_lower),
                    "bb_mid":   float(bb_mid),
                    "atr":      float(current_atr),
                    "signals":  signals,
                },
                "ohlc_data": json.loads(
                    hist_daily.to_json(orient="split", date_format="iso")
                ),
                "reasoning": ai_reasoning,
                "hyperparams_source": hyperparams_source,
            }

        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Portfolio Optimizer — Riskfolio-Lib
# Supports: HRP, CVaR, Mean-Variance (Sharpe)
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    @staticmethod
    def optimize(tickers, risk_tolerance):
        """
        risk_tolerance 0.0 – 0.33 → HRP  (Hierarchical Risk Parity)
        risk_tolerance 0.34 – 0.66 → Min CVaR (Conditional Value at Risk)
        risk_tolerance 0.67 – 1.0  → Max Sharpe (Mean-Variance)
        Falls back to equal-weight if data is insufficient.
        """
        try:
            frames = {}
            for t in tickers:
                hist = get_price_history(t, period="2y")
                if hist is not None and not hist.empty:
                    frames[t] = hist["Close"]

            if len(frames) < 2:
                raise ValueError("Need at least 2 tickers with data.")

            prices  = pd.DataFrame(frames).dropna()
            if len(prices) < 60:
                raise ValueError("Not enough historical data.")

            returns = prices.pct_change().dropna()

            if risk_tolerance < 0.34:
                # HRP — best diversification, no matrix inversion needed
                hrp     = rp.HCPortfolio(returns=returns)
                w       = hrp.optimization(
                    model="HRP", codependence="pearson",
                    rm="MV", rf=0, linkage="ward",
                    max_k=10, leaf_order=True
                )
                weights = w["weights"].to_dict()

            elif risk_tolerance < 0.67:
                # Minimum CVaR — controls tail risk
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                w = port.optimization(
                    model="Classic", rm="CVaR", obj="MinRisk",
                    rf=0.05, l=0, hist=True
                )
                weights = w["weights"].to_dict()

            else:
                # Maximum Sharpe — aggressive growth
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                w = port.optimization(
                    model="Classic", rm="MV", obj="Sharpe",
                    rf=0.05, l=0, hist=True
                )
                weights = w["weights"].to_dict()

            return {k: round(float(v), 4) for k, v in weights.items() if float(v) > 0.001}

        except Exception:
            n = len(tickers)
            return {t: round(1.0 / n, 4) for t in tickers}
