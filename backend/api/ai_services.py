import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from groq import Groq
import yfinance as yf

try:
    _YF_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "yfinance"
    _YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_YF_CACHE_DIR))
except Exception:
    pass

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
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        # LayerNorm over the concatenated forward+backward hidden states
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        # Fixed 0.2 output dropout for regularisation (independent of LSTM inter-layer dropout)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden  = self.layer_norm(lstm_out[:, -1, :])
        return self.fc(self.dropout(last_hidden))


def _directional_penalty_loss(pred: torch.Tensor, target: torch.Tensor,
                               penalty_weight: float = 0.5) -> torch.Tensor:
    """
    Loss = MSE(pred, target)  +  penalty_weight * fraction_of_directional_mistakes.
    Works on log-return tensors: same sign → agreement, opposite sign → penalised.
    """
    mse = torch.mean((pred - target) ** 2)
    # sign(pred)*sign(target): +1 = directions agree, -1 = disagree
    sign_agreement = torch.sign(pred) * torch.sign(target)
    # clamp so only disagreements (negative values) contribute to penalty
    penalty = torch.mean(torch.clamp(-sign_agreement, min=0.0))
    return mse + penalty_weight * penalty


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
                            "You are a helpful financial assistant for the Stock Price Predictor app. "
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

def get_price_history(symbol, period="6mo", target_date=None):
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
            
            if target_date:
                end_date = pd.to_datetime(target_date)
            else:
                end_date = datetime.now()
            
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

    # yfinance fallback.
    try:
        if target_date:
            from datetime import timedelta
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
            days = period_days.get(period, 180)
            end_dt = pd.to_datetime(target_date)
            start_dt = end_dt - timedelta(days=days)
            # Use auto_adjust=True for consistent historical benchmarking (Adjusted Close)
            hist = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"), end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, threads=False, auto_adjust=True)
        else:
            hist = yf.download(symbol, period=period, progress=False, threads=False, auto_adjust=True)
            
        if isinstance(hist, pd.DataFrame) and isinstance(hist.columns, pd.MultiIndex):
            try:
                hist = hist.droplevel(-1, axis=1)
            except Exception:
                hist.columns = hist.columns.get_level_values(0)
        return hist if hist is not None and not hist.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_price_window(symbol, start_date, end_date):
    """Fetch price history between two specific dates."""
    if polygon_client:
        try:
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
                    "Open":   a.open, "High": a.high, "Low": a.low, "Close": a.close, "Volume": a.volume,
                } for a in aggs], index=pd.to_datetime([a.timestamp for a in aggs], unit="ms"))
                if not df.empty: return df
        except Exception: pass
    
    try:
        # Backtesting uses auto_adjust=True to match historical charts
        hist = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False, threads=False, auto_adjust=True)
        if isinstance(hist, pd.DataFrame) and isinstance(hist.columns, pd.MultiIndex):
            hist = hist.droplevel(-1, axis=1)
        return hist
    except Exception: return pd.DataFrame()


def get_ticker_info(symbol, target_date=None):
    """
    Fetch company info.
    Primary:  Polygon.io
    Fallback: yfinance
    If target_date is provided, we fetch historical price to avoid "future" currentPrice leak.
    """
    info = {}
    if polygon_client:
        try:
            detail = polygon_client.get_ticker_details(symbol.upper())
            if detail:
                info = {
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

    if not info:
        try:
            info = yf.Ticker(symbol).info
        except Exception:
            info = {}

    # Crucial for backtesting: override currentPrice with historical data
    if target_date:
        hist = get_price_history(symbol, period="5d", target_date=target_date)
        if hist is not None and not hist.empty:
            close = TrendPredictor._get_close_series(hist)
            if not close.empty:
                # Use the exact price from the target date (Adjusted Close)
                info["currentPrice"] = float(close.iloc[-1])
                info["regularMarketPrice"] = float(close.iloc[-1]) # Support both yfinance keys
    
    return info


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
    def _get_close_series(price_frame):
        if price_frame is None or getattr(price_frame, "empty", True):
            return pd.Series(dtype=float)

        close = price_frame["Close"] if "Close" in price_frame else price_frame
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                close = close.squeeze()

        close = pd.to_numeric(close, errors="coerce").dropna()
        close.name = "Close"
        return close

    @staticmethod
    def _select_prediction_benchmark(symbol):
        return "^NSEI" if symbol.endswith((".NS", ".BO")) else "SPY"

    @staticmethod
    def _historical_mean_projection(symbol, lookback_days=252, target_date=None):
        """
        Simple and explicit fallback when no reliable LSTM is available:
        use the mean daily excess return over the last 252 trading days,
        anchored to a broad market benchmark.
        """
        hist = get_price_history(symbol, period="2y", target_date=target_date)
        close = TrendPredictor._get_close_series(hist)
        if close.empty or len(close) < 20:
            return None

        current_price = float(close.iloc[-1])
        asset_returns = close.pct_change().dropna()
        if asset_returns.empty:
            return None

        benchmark_symbol = TrendPredictor._select_prediction_benchmark(symbol)
        benchmark_close = TrendPredictor._get_close_series(
            get_price_history(benchmark_symbol, period="2y", target_date=target_date)
        )
        benchmark_returns = benchmark_close.pct_change().dropna() if not benchmark_close.empty else pd.Series(dtype=float)

        aligned_asset = asset_returns
        aligned_benchmark = benchmark_returns
        benchmark_used = False

        if not benchmark_returns.empty:
            aligned_asset, aligned_benchmark = asset_returns.align(
                benchmark_returns,
                join="inner",
            )
            benchmark_used = len(aligned_asset) >= 20

        sample_size = min(len(aligned_asset), lookback_days)
        if sample_size < 20:
            sample_size = min(len(asset_returns), lookback_days)
            asset_window = asset_returns.tail(sample_size)
            mean_daily_return = float(asset_window.mean())
            mean_daily_excess = None
            benchmark_daily_return = None
            benchmark_used = False
        else:
            asset_window = aligned_asset.tail(sample_size)
            benchmark_window = aligned_benchmark.tail(sample_size)
            mean_daily_excess = float((asset_window - benchmark_window).mean())
            benchmark_daily_return = float(benchmark_window.mean())
            mean_daily_return = mean_daily_excess + benchmark_daily_return

        mean_daily_return = float(np.clip(mean_daily_return, -0.25, 0.25))
        projected_price = current_price * (1.0 + mean_daily_return)
        label = (
            f"Historical mean daily excess return vs {benchmark_symbol} ({sample_size}d)"
            if benchmark_used
            else f"Historical mean daily return ({sample_size}d)"
        )

        return {
            "predicted_price": float(projected_price),
            "expected_return": float(mean_daily_return),
            "prediction_method": "historical_mean_fallback",
            "prediction_label": label,
            "benchmark_symbol": benchmark_symbol if benchmark_used else None,
            "mean_daily_excess": mean_daily_excess,
            "benchmark_daily_return": benchmark_daily_return,
            "lookback_days": int(sample_size),
        }

    @staticmethod
    def forecast_price(
        symbol,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        epochs=100,  # Increased intensity for accuracy
        dropout=0.1,  # Slight dropout for better generalization
        seq_len=20,
        allow_lstm=True,
        target_date=None,
    ):
        if allow_lstm:
            lstm_price = TrendPredictor._lstm_predict(
                symbol,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
                epochs=epochs,
                dropout=dropout,
                seq_len=seq_len,
                target_date=target_date,
            )

            latest_close = TrendPredictor._get_close_series(
                get_price_history(symbol, period="5d", target_date=target_date)
            )
            if (
                lstm_price is not None
                and np.isfinite(lstm_price)
                and lstm_price > 0
                and not latest_close.empty
            ):
                current_price = float(latest_close.iloc[-1])
                expected_return = float((lstm_price - current_price) / current_price)
                return {
                    "predicted_price": float(lstm_price),
                    "expected_return": expected_return,
                    "prediction_method": "lstm",
                    "prediction_label": "Evolved LSTM next-day forecast",
                    "benchmark_symbol": None,
                    "mean_daily_excess": None,
                    "benchmark_daily_return": None,
                    "lookback_days": None,
                }

        return TrendPredictor._historical_mean_projection(symbol, target_date=target_date)

    @staticmethod
    def _lstm_predict(symbol, hidden_size=64, num_layers=2, learning_rate=0.001, epochs=100, dropout=0.1, seq_len=20, target_date=None):
        """
        Trains a Bidirectional LSTM to predict weekly log returns, with:
          - Stationary log-return target (eliminates price-level non-stationarity)
          - Early stopping on a held-out 10 % validation slice (patience=10)
          - ReduceLROnPlateau scheduler (patience=5, factor=0.5)
          - Directional penalty loss
        Returns the next-period predicted price as an absolute float, or None on failure.
        """
        try:
            import ta

            df = get_price_history(symbol, period="2y", target_date=target_date)
            if df is None or getattr(df, 'empty', True):
                return None

            df = df.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()

            close = df['Close']
            high  = df['High']
            low   = df['Low']

            df['rsi']       = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            df['macd_diff'] = ta.trend.MACD(close=close).macd_diff()
            df['bb_width']  = (
                ta.volatility.BollingerBands(close=close, window=20).bollinger_hband() -
                ta.volatility.BollingerBands(close=close, window=20).bollinger_lband()
            )
            df['atr']       = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
            sma20           = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
            df['sma_dist']  = (close - sma20) / sma20
            df['obv']       = ta.volume.OnBalanceVolumeIndicator(
                close=df['Close'], volume=df['Volume']
            ).on_balance_volume()

            feature_cols = ['Close', 'rsi', 'macd_diff', 'bb_width', 'atr', 'sma_dist', 'obv']
            df = df[feature_cols].dropna()
            if len(df) < seq_len + 10:
                return None

            # Stationary target: log returns of Close
            log_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values

            scaler      = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(df.values)
            input_size  = data_scaled.shape[1]

            X, y = [], []
            for i in range(len(data_scaled) - seq_len):
                X.append(data_scaled[i:i + seq_len])
                y.append(log_returns[i + seq_len])

            if not X:
                return None

            X = np.array(X)
            y = np.array(y).reshape(-1, 1)

            train_size = int(len(X) * 0.8)
            val_split  = max(1, int(train_size * 0.1))  # last 10 % of train for early stopping

            X_train_t    = torch.tensor(X[:train_size], dtype=torch.float32)
            y_train_t    = torch.tensor(y[:train_size], dtype=torch.float32)
            X_train_core = X_train_t[:-val_split]
            y_train_core = y_train_t[:-val_split]
            X_val_es     = X_train_t[-val_split:]
            y_val_es     = y_train_t[-val_split:]

            model     = LSTMModel(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, dropout=dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )

            best_val_loss    = float('inf')
            patience_counter = 0
            best_state       = None
            PATIENCE         = 10

            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                out  = model(X_train_core)
                loss = _directional_penalty_loss(out, y_train_core)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_out  = model(X_val_es)
                    val_loss = torch.mean((val_out - y_val_es) ** 2).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break

            if best_state:
                model.load_state_dict(best_state)

            # Predict next period's log return from the most recent full window
            model.eval()
            with torch.no_grad():
                next_seq     = torch.tensor(
                    data_scaled[-seq_len:].reshape(1, seq_len, input_size),
                    dtype=torch.float32
                )
                pred_log_ret = model(next_seq).item()

            # Reconstruct absolute price: last known Close × exp(predicted log return)
            last_close = float(df['Close'].iloc[-1])
            pred_price = last_close * float(np.exp(pred_log_ret))
            return pred_price if np.isfinite(pred_price) and pred_price > 0 else None

        except Exception:
            return None

    @staticmethod
    def predict(symbol, hidden_size=64, num_layers=2, learning_rate=0.001, epochs=30, dropout=0.0, seq_len=20, hyperparams_source='default', target_date=None):
        try:
            # ---- Price History ----
            hist_daily  = get_price_history(symbol, period="6mo", target_date=target_date)
            hist_weekly = get_price_history(symbol, period="1y", target_date=target_date)
            if hist_weekly is not None and not hist_weekly.empty:
                hist_weekly = hist_weekly.resample("W").last().tail(30)

            if hist_daily is None or hist_daily.empty:
                return {"error": f"No data found for {symbol}."}

            # ---- Company Info ----
            info    = get_ticker_info(symbol, target_date=target_date)
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
            # For backtesting integrity, we do not show live news if target_date is in the past
            news = []
            if not target_date:
                try:
                    ticker   = yf.Ticker(symbol)
                    raw_news = ticker.news[:3] if ticker.news else []
                    news     = TrendPredictor._parse_news(raw_news)
                except Exception:
                    news = []
            news_text = "\n".join([f"- {n['title']} ({n['publisher']})" for n in news]) if news else "News unavailable for historical backtesting to ensure model integrity."

            # ---- Peers ----
            peers        = TrendPredictor.get_peers(symbol, sector)
            peer_data    = []
            peer_history = {}

            target_data           = financials.copy()
            target_data["Symbol"] = symbol
            peer_data.append(target_data)
            peer_history[symbol]  = TrendPredictor._get_close_series(hist_daily).tolist()

            for p in peers:
                try:
                    p_hist = get_price_history(p, period="6mo", target_date=target_date)
                    p_info = get_ticker_info(p)
                    p_fin  = extract_financials(p_info)
                    p_fin["Symbol"] = p
                    peer_data.append(p_fin)
                    if p_hist is not None and not p_hist.empty:
                        peer_history[p] = TrendPredictor._get_close_series(p_hist).tolist()
                except Exception:
                    continue

            # ---- Technical Indicators ----
            if len(hist_daily) < 50:
                return {"error": "Not enough historical data for technical analysis."}

            close  = TrendPredictor._get_close_series(hist_daily)

            def _get_series(df, col):
                if col in df:
                    s = df[col]
                    if isinstance(s, pd.DataFrame):
                        return s.iloc[:, 0] if s.shape[1] >= 1 else s.squeeze()
                    return s
                return pd.Series(dtype=float)

            high   = _get_series(hist_daily, "High")
            low    = _get_series(hist_daily, "Low")
            volume = _get_series(hist_daily, "Volume")

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

            forecast = TrendPredictor.forecast_price(
                symbol,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
                epochs=epochs,
                dropout=dropout,
                seq_len=seq_len,
                allow_lstm=(hyperparams_source == "evolved"),
                target_date=target_date,
            )
            if forecast:
                ret = forecast.get("expected_return", 0.0)
                # realism: Dampened Momentum logic. 
                # Instead of raw compounding (1+r)^30, we use a logistic decay 
                # because market signals mean-revert and don't stay linear forever.
                import math
                def dampen_ret(r, days, half_life=10):
                    # Signal decays over time toward zero
                    total_ret = 0
                    current_r = r
                    for _ in range(days):
                        total_ret += current_r
                        current_r *= (0.5 ** (1/half_life)) # exponential decay of signal
                    return total_ret

                if forecast.get("prediction_method") == "lstm":
                    # LSTM return is already weekly or next-period; we adjust it
                    ret_30d = dampen_ret(ret / 5.0, 30, half_life=15)
                    prediction_label = forecast.get("prediction_label", "") + " (30-day Dampened)"
                else:
                    # Historical mean is very prone to being too lucky
                    ret_30d = dampen_ret(ret, 30, half_life=7)
                    prediction_label = forecast.get("prediction_label", "") + " (30-day Dampened)"
                
                predicted_price = float(recent_close * (1.0 + ret_30d))
                prediction_method = forecast.get("prediction_method", "") + "_dampened"
            else:
                predicted_price = float(recent_close)
                prediction_method = "technical_signal_fallback"
                prediction_label = "Technical signal fallback"

            if predicted_price is not None:
                if predicted_price > recent_close:
                    trend = "UP"
                elif predicted_price < recent_close:
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

            last_30_days_prices = TrendPredictor._get_close_series(hist_daily).tail(30).tolist()

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

            # ---- Backtest Verification (Success Discovery) ----
            backtest_verification = None
            if target_date:
                try:
                    from datetime import timedelta
                    start_ver = pd.to_datetime(target_date) + timedelta(days=1)
                    end_ver   = start_ver + timedelta(days=30)
                    # Check if end_ver is in the past
                    if end_ver < pd.to_datetime("today"):
                        ver_hist = get_price_window(symbol, start_ver, end_ver)
                        if ver_hist is not None and not ver_hist.empty:
                            ver_close = TrendPredictor._get_close_series(ver_hist)
                            if not ver_close.empty:
                                actual_final_price = float(ver_close.iloc[-1])
                                actual_ret = (actual_final_price - recent_close) / recent_close
                                backtest_verification = {
                                    "actual_price_30d": actual_final_price,
                                    "actual_return_30d": actual_ret,
                                    "days_data": len(ver_close),
                                }
                except Exception:
                    pass

            return {
                "symbol":          symbol,
                "current_price":   float(recent_close),
                "predicted_price": float(predicted_price),
                "trend":           trend,
                "financials":      financials,
                "news":            news,
                "peer_financials": peer_data,
                "peer_history":    peer_history,
                "weekly_data":     TrendPredictor._get_close_series(hist_weekly).tolist() if hist_weekly is not None and getattr(hist_weekly, 'empty', False) is False else [],
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
                "prediction_method": prediction_method,
                "prediction_label": prediction_label,
                "backtest_verification": backtest_verification,
            }

        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def run_backtest(cls, symbol, past_date_str, force_retrain=False):
        from .models import OptimizedHyperparams
        import ta
        from datetime import datetime

        df = get_price_history(symbol, period="5y")
        if df is None or getattr(df, 'empty', True):
            return {"error": f"Not enough data for {symbol}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Resample logic matches Model Evolution logic natively
        df = df.resample('W').agg({
            'Open':   'first',
            'High':   'max',
            'Low':    'min',
            'Close':  'last',
            'Volume': 'sum'
        }).dropna()
        
        close = df['Close']
        high  = df['High']
        low   = df['Low']
        
        df['rsi']        = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df['macd_diff']  = ta.trend.MACD(close=close).macd_diff()
        df['bb_width']   = (
            ta.volatility.BollingerBands(close=close, window=20).bollinger_hband() -
            ta.volatility.BollingerBands(close=close, window=20).bollinger_lband()
        )
        df['atr']        = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        sma20            = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        df['sma_dist']   = (close - sma20) / sma20
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()

        # Save Open prices before dropping to feature columns (needed for T+1 execution)
        open_series = df['Open'].copy()

        feature_cols = ['Close', 'rsi', 'macd_diff', 'bb_width', 'atr', 'sma_dist', 'obv']
        df = df[feature_cols].dropna()

        # Align open_series to the clean df index (after indicator-driven dropna)
        open_series = open_series.reindex(df.index)
        
        past_date = pd.to_datetime(past_date_str)
        if past_date.tzinfo is None and df.index.tzinfo is not None:
            past_date = past_date.tz_localize(df.index.tzinfo)
            
        if force_retrain:
            # Purge stale cache so future standard predictions also use fresh weights
            try:
                OptimizedHyperparams.objects.filter(symbol=symbol.upper()).delete()
            except Exception:
                pass
            cached = None
        else:
            cached = OptimizedHyperparams.get_valid_cache(symbol)

        seq_len     = cached.seq_len        if cached else 20
        hidden_size = cached.hidden_size    if cached else 64
        num_layers  = cached.num_layers     if cached else 2
        lr          = cached.learning_rate  if cached else 0.001
        epochs      = cached.epochs         if cached else 50   # bump default for fresh runs
        dropout     = cached.dropout        if cached else 0.2  # match new architecture default
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df.values)
        input_size  = data_scaled.shape[1]

        # Stationary target: log returns of Close (eliminates price-level non-stationarity)
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values

        X, y, target_dates = [], [], []
        for i in range(len(data_scaled) - seq_len):
            X.append(data_scaled[i:i + seq_len])
            y.append(log_returns[i + seq_len])   # log return, not scaled price
            target_dates.append(df.index[i + seq_len])
            
        if not X:
            return {"error": "Not enough data after applying sequence length."}
            
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split Train/Test
        train_idx = [i for i, d in enumerate(target_dates) if d <= past_date]
        test_idx = [i for i, d in enumerate(target_dates) if d > past_date]
        
        if len(train_idx) < 10:
            return {"error": "Past date is too early. Not enough training data."}
        if not test_idx:
            return {"error": "Past date is too recent. No test data available."}
            
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Deterministic seed — must run unconditionally so every backtest run is
        # perfectly reproducible regardless of the force_retrain flag.
        import random as _random
        torch.manual_seed(42)
        np.random.seed(42)
        _random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Train — with early stopping, ReduceLROnPlateau, and directional penalty loss
        model     = LSTMModel(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)

        # Carve last 10 % of train slice as the early-stopping validation set
        val_split    = max(1, int(len(X_train_t) * 0.1))
        X_tr_core    = X_train_t[:-val_split]
        y_tr_core    = y_train_t[:-val_split]
        X_val_es     = X_train_t[-val_split:]
        y_val_es     = y_train_t[-val_split:]

        best_val_loss    = float('inf')
        patience_counter = 0
        best_state       = None
        PATIENCE         = 10

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            out  = model(X_tr_core)
            loss = _directional_penalty_loss(out, y_tr_core)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out  = model(X_val_es)
                val_loss = torch.mean((val_out - y_val_es) ** 2).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Predict log returns on test sequences
        model.eval()
        with torch.no_grad():
            X_test_t     = torch.tensor(X_test, dtype=torch.float32)
            pred_log_rets = model(X_test_t).numpy().flatten()

        # Previous close prices (last scaled Close in each input window → unscaled)
        # These are needed to reconstruct absolute prices from log returns.
        prev_scaled_raw = X_test[:, -1, 0]
        dummy_prev      = np.zeros((len(prev_scaled_raw), input_size))
        dummy_prev[:, 0] = prev_scaled_raw
        prevs = scaler.inverse_transform(dummy_prev)[:, 0]

        # Reconstruct absolute prices: price(T) = price(T-1) × exp(log_return(T))
        y_test_flat = y_test.flatten()
        actuals = prevs * np.exp(y_test_flat)   # actual close prices
        preds   = prevs * np.exp(pred_log_rets) # predicted close prices
        
        # Trading Friction constants
        TRANSACTION_FEE = 0.001   # 0.1% commission per trade
        SLIPPAGE        = 0.0005  # 0.05% slippage per trade

        from sklearn.metrics import mean_squared_error, mean_absolute_error

        # prevs already computed above via log-return price reconstruction
        predicted_deltas = preds - prevs
        predicted_signs  = np.sign(predicted_deltas)

        # T+1 Open Execution: build execution prices for each test step
        execution_opens = []
        for enum_k, i in enumerate(test_idx):
            date_i = target_dates[i]
            loc = df.index.get_loc(date_i)
            if loc + 1 < len(df.index):
                execution_opens.append(float(open_series.iloc[loc + 1]))
            else:
                execution_opens.append(float(actuals[enum_k]))  # fallback: last close
        execution_opens = np.array(execution_opens)

        # Error metrics measured against T+1 execution prices (not same-day close)
        rmse = float(np.sqrt(mean_squared_error(execution_opens, preds)))
        mae  = float(mean_absolute_error(execution_opens, preds))

        # Directional accuracy: did predicted direction match T+1 open vs prev close?
        actual_deltas_exec = execution_opens - prevs
        actual_signs_exec  = np.sign(actual_deltas_exec)
        correct_exec = np.sum(actual_signs_exec == predicted_signs)
        dir_acc = (correct_exec / len(actual_signs_exec)) * 100 if len(actual_signs_exec) > 0 else 0.0

        # Expected Realized Return (Net of Fees)
        # Strategy: at each step follow model signal, enter at Open(T+1), exit at Open(T+2)
        net_return = 0.0
        num_trades = 0
        for k in range(len(execution_opens) - 1):
            signal = predicted_signs[k]
            if signal == 0:
                continue
            entry      = execution_opens[k]
            exit_price = execution_opens[k + 1]
            if entry <= 0:
                continue
            raw_pct     = signal * (exit_price - entry) / entry
            net_return += raw_pct - TRANSACTION_FEE - SLIPPAGE
            num_trades += 1
        expected_realized_return = float(net_return * 100) if num_trades > 0 else 0.0

        # Extract Historical context
        hist_df = df[df.index <= past_date]['Close']
        if hist_df.empty:
            hist_dates = []
            hist_prices = []
        else:
            hist_dates = [d.strftime('%Y-%m-%d') for d in hist_df.index]
            hist_prices = hist_df.tolist()

        test_dates_str = [target_dates[i].strftime('%Y-%m-%d') for i in test_idx]

        # Per-row trade log
        trade_log = []
        for k in range(len(test_idx)):
            trade_log.append({
                "date":              test_dates_str[k],
                "predicted_price":   round(float(preds[k]), 2),
                "actual_price":      round(float(actuals[k]), 2),
                "error_abs":         round(float(abs(preds[k] - actuals[k])), 2),
                "error_pct":         round(float(abs(preds[k] - actuals[k]) / actuals[k] * 100), 2),
                "direction_correct": bool(predicted_signs[k] == actual_signs_exec[k]),
            })

        # Win rate
        win_rate = float(
            sum(1 for r in trade_log if r["direction_correct"]) / len(trade_log) * 100
        ) if trade_log else 0.0

        # Equity curves (normalized to 100 at start)
        lstm_equity = [100.0]
        buyhold_equity = [100.0]
        for k in range(len(execution_opens) - 1):
            signal = predicted_signs[k]
            entry  = execution_opens[k]
            exit_p = execution_opens[k + 1]
            if entry > 0 and signal != 0:
                raw_ret = signal * (exit_p - entry) / entry
                net_ret = raw_ret - TRANSACTION_FEE - SLIPPAGE
            else:
                net_ret = 0.0
            lstm_equity.append(lstm_equity[-1] * (1 + net_ret))

            bh_ret = (exit_p - entry) / entry if entry > 0 else 0.0
            buyhold_equity.append(buyhold_equity[-1] * (1 + bh_ret))

        equity_dates = test_dates_str[:len(lstm_equity)]

        # Sharpe (annualized, weekly data → sqrt(52))
        lstm_returns = np.diff(lstm_equity) / np.array(lstm_equity[:-1])
        if len(lstm_returns) > 1 and lstm_returns.std() > 0:
            sharpe = float((lstm_returns.mean() / lstm_returns.std()) * np.sqrt(52))
        else:
            sharpe = 0.0

        # Max drawdown
        peak = lstm_equity[0]
        max_dd = 0.0
        for val in lstm_equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        max_drawdown = float(max_dd * 100)

        return {
            "historical": {"dates": hist_dates, "prices": hist_prices},
            "test": {
                "dates": test_dates_str,
                "actual": actuals.tolist(),
                "predicted": preds.tolist()
            },
            "metrics": {
                "rmse": rmse,
                "mae": mae,
                "directional_accuracy": float(dir_acc),
                "expected_realized_return": expected_realized_return,
                "win_rate":      round(win_rate, 1),
                "sharpe_ratio":  round(sharpe, 3),
                "max_drawdown":  round(max_drawdown, 2),
            },
            "trade_log": trade_log,
            "equity_curve": {
                "dates":          equity_dates,
                "lstm_equity":    [round(v, 4) for v in lstm_equity],
                "buyhold_equity": [round(v, 4) for v in buyhold_equity],
            },
        }

# ---------------------------------------------------------------------------
# Portfolio Optimizer — Riskfolio-Lib
# Supports: HRP, CVaR, Mean-Variance (Sharpe)
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    MAX_SINGLE_WEIGHT = 0.60
    MIN_BREADTH = 3
    MIN_POSITION_WEIGHT = 0.01

    @staticmethod
    def _extract_weight_series(weights_df, tickers):
        if weights_df is None or getattr(weights_df, "empty", True):
            raise ValueError("Optimizer returned no weights.")

        if isinstance(weights_df, pd.Series):
            series = weights_df.astype(float)
        elif "weights" in weights_df.columns:
            series = weights_df["weights"].astype(float)
        else:
            series = weights_df.iloc[:, 0].astype(float)

        series = series.reindex(tickers).fillna(0.0)
        return series

    @staticmethod
    def _normalize_weight_series(weight_series, tickers):
        series = pd.Series(weight_series, dtype=float).reindex(tickers).fillna(0.0)
        series = series.clip(lower=0.0)
        total = float(series.sum())
        if total <= 1e-10:
            return pd.Series({t: 1.0 / len(tickers) for t in tickers})

        return series / total

    @staticmethod
    def _inverse_volatility_weights(cov_matrix, tickers):
        """Inverse-volatility weighting from a covariance matrix."""
        vols = pd.Series(np.sqrt(np.diag(cov_matrix.values)), index=cov_matrix.index)
        inv_vols = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return PortfolioOptimizer._normalize_weight_series(inv_vols, tickers)

    @staticmethod
    def _positive_return_weights(expected_returns_series, tickers):
        """Weight proportional to positive expected returns only."""
        pos = pd.Series(expected_returns_series, dtype=float).clip(lower=0.0)
        return PortfolioOptimizer._normalize_weight_series(pos.reindex(tickers).fillna(0.0), tickers)

    @staticmethod
    def _enforce_min_breadth(weight_series, tickers):
        series = PortfolioOptimizer._normalize_weight_series(weight_series, tickers)

        if len(tickers) < PortfolioOptimizer.MIN_BREADTH:
            return series

        top_tickers = list(series.sort_values(ascending=False).index[:PortfolioOptimizer.MIN_BREADTH])
        adjusted = series.copy()

        for ticker in top_tickers:
            adjusted.loc[ticker] = max(
                float(adjusted.loc[ticker]),
                PortfolioOptimizer.MIN_POSITION_WEIGHT,
            )

        min_allowed = pd.Series(0.0, index=adjusted.index)
        min_allowed.loc[top_tickers] = PortfolioOptimizer.MIN_POSITION_WEIGHT
        excess = float(adjusted.sum() - 1.0)

        if excess > 1e-10:
            headroom = (adjusted - min_allowed).clip(lower=0.0)
            headroom_total = float(headroom.sum())
            if headroom_total <= 1e-10:
                return PortfolioOptimizer._normalize_weight_series(series, tickers)
            adjusted -= headroom / headroom_total * excess

        return PortfolioOptimizer._normalize_weight_series(adjusted, tickers)

    @staticmethod
    def _apply_max_weight_cap(weight_series, tickers):
        series = PortfolioOptimizer._normalize_weight_series(weight_series, tickers)
        max_weight = PortfolioOptimizer.MAX_SINGLE_WEIGHT

        for _ in range(len(tickers) + 2):
            over_cap = series > max_weight + 1e-10
            if not over_cap.any():
                break

            capped = series.copy()
            capped.loc[over_cap] = max_weight
            excess = float(1.0 - capped.sum())
            if excess <= 1e-10:
                series = PortfolioOptimizer._normalize_weight_series(capped, tickers)
                continue

            under_cap = capped < max_weight - 1e-10
            if not under_cap.any():
                series = PortfolioOptimizer._normalize_weight_series(capped, tickers)
                break

            redistribution_base = series.loc[under_cap].clip(lower=0.0)
            redistribution_total = float(redistribution_base.sum())
            if redistribution_total <= 1e-10:
                redistribution_base = pd.Series(
                    1.0,
                    index=capped.index[under_cap],
                    dtype=float,
                )
                redistribution_total = float(redistribution_base.sum())

            capped.loc[under_cap] += redistribution_base / redistribution_total * excess
            series = PortfolioOptimizer._normalize_weight_series(capped, tickers)

        return series.clip(upper=max_weight)

    @staticmethod
    def _stabilize_weight_series(weight_series, tickers):
        series = PortfolioOptimizer._normalize_weight_series(weight_series, tickers)
        series = PortfolioOptimizer._enforce_min_breadth(series, tickers)
        series = PortfolioOptimizer._apply_max_weight_cap(series, tickers)
        series = PortfolioOptimizer._enforce_min_breadth(series, tickers)
        return PortfolioOptimizer._normalize_weight_series(series, tickers)

    @staticmethod
    def _build_result(
        requested_tickers,
        valid_tickers,
        allocation,
        source,
        blend_meta=None,
        fallback_reason=None,
    ):
        ordered_allocation = {
            ticker: round(float(pd.Series(allocation, dtype=float).reindex(valid_tickers).fillna(0.0).loc[ticker]), 4)
            for ticker in valid_tickers
        }
        return {
            "allocation": ordered_allocation,
            "requested_tickers": requested_tickers,
            "valid_tickers": valid_tickers,
            "dropped_tickers": [ticker for ticker in requested_tickers if ticker not in valid_tickers],
            "source": source,
            "blend_meta": blend_meta or {},
            "fallback_reason": fallback_reason,
            "constraints": {
                "max_single_weight": PortfolioOptimizer.MAX_SINGLE_WEIGHT,
                "min_positions_over_one_pct": (
                    PortfolioOptimizer.MIN_BREADTH
                    if len(valid_tickers) >= PortfolioOptimizer.MIN_BREADTH
                    else len(valid_tickers)
                ),
            },
        }

    @staticmethod
    def _blend_profiles(conservative, balanced, aggressive, risk_tolerance):
        risk_tolerance = float(np.clip(risk_tolerance, 0.0, 1.0))
        if risk_tolerance <= 0.5:
            alpha = risk_tolerance / 0.5
            blended = conservative * (1.0 - alpha) + balanced * alpha
            blend_meta = {"from": "HRP", "to": "Min CVaR", "alpha": round(alpha, 4)}
        else:
            alpha = (risk_tolerance - 0.5) / 0.5
            blended = balanced * (1.0 - alpha) + aggressive * alpha
            blend_meta = {"from": "Min CVaR", "to": "Max Sharpe", "alpha": round(alpha, 4)}
        return blended, blend_meta

    @staticmethod
    def optimize(tickers, risk_tolerance, target_date=None):
        """
        Builds conservative, balanced, and aggressive portfolios, then blends
        them continuously so allocations respond smoothly to risk_tolerance.
        """
        requested_tickers = [str(t).upper() for t in tickers]
        try:
            from pypfopt import EfficientFrontier, expected_returns, risk_models

            frames = {}
            for t in requested_tickers:
                hist = get_price_history(t, period="2y", target_date=target_date)
                if hist is not None and not hist.empty:
                    close = TrendPredictor._get_close_series(hist)
                    if not close.empty:
                        frames[t] = close

            if len(frames) < 2:
                raise ValueError("Need at least 2 tickers with data.")

            prices  = pd.DataFrame(frames).dropna()
            if len(prices) < 60:
                raise ValueError("Not enough historical data.")

            daily_returns = prices.pct_change().dropna()
            valid_tickers = list(daily_returns.columns)
            
            # Robust return estimation: use Exponential Moving Average (EMA) 
            # for expected returns to better capture recent momentum up to the target_date
            mu = expected_returns.ema_historical_return(prices, span=180) 
            cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

            # Conservative: inverse volatility weighting.
            vols = daily_returns.std().replace(0, np.nan)
            conservative = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            conservative = PortfolioOptimizer._normalize_weight_series(conservative, valid_tickers)

            # Balanced: minimum volatility portfolio.
            ef_min_vol = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef_min_vol.min_volatility()
            balanced = PortfolioOptimizer._extract_weight_series(
                pd.Series(ef_min_vol.clean_weights()),
                valid_tickers,
            )

            # Aggressive: maximum Sharpe, with a stable fallback.
            ef_max_sharpe = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            try:
                ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
            except Exception:
                ef_max_sharpe.max_quadratic_utility(risk_aversion=0.5)
            aggressive = PortfolioOptimizer._extract_weight_series(
                pd.Series(ef_max_sharpe.clean_weights()),
                valid_tickers,
            )

            blended, blend_meta = PortfolioOptimizer._blend_profiles(
                conservative,
                balanced,
                aggressive,
                risk_tolerance,
            )
            cleaned = PortfolioOptimizer._stabilize_weight_series(blended, valid_tickers)
            return PortfolioOptimizer._build_result(
                requested_tickers=requested_tickers,
                valid_tickers=valid_tickers,
                allocation=cleaned,
                source="risk_based",
                blend_meta=blend_meta,
                fallback_reason=None,
            )

        except Exception as exc:
            valid_tickers = requested_tickers[:]
            if not valid_tickers:
                valid_tickers = ["SPY"]
            fallback_weights = pd.Series(
                {ticker: 1.0 / len(valid_tickers) for ticker in valid_tickers},
                dtype=float,
            )
            return PortfolioOptimizer._build_result(
                requested_tickers=requested_tickers or valid_tickers,
                valid_tickers=valid_tickers,
                allocation=fallback_weights,
                source="risk_based",
                blend_meta={},
                fallback_reason=str(exc),
            )


import hashlib

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
        class _GymStub:
            class Env: pass
            class spaces:
                class Box: pass
        gym = _GymStub()
        spaces = _GymStub.spaces

try:
    from stable_baselines3 import A2C, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    class BaseCallback:
        def __init__(self, *args, **kwargs): pass
        def _on_step(self): return True

MAX_SINGLE_WEIGHT  = 0.45
TRANSACTION_COST   = 0.001
RISK_FREE_RATE_ANN = 0.05

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


def compute_fundamental_score(info):
    scores = []
    for value, scale, invert in [
        (info.get("trailingPE"), 50.0, True),
        (info.get("debtToEquity"), 200.0, True),
        (info.get("revenueGrowth"), 0.20, False),
    ]:
        try:
            v = float(value)
            scores.append(float(np.clip(1.0 - v / scale if invert else v / scale, 0.0, 1.0)))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores))


def fetch_ticker_features(tickers):
    features = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "")
            fund_score = compute_fundamental_score(info)
            headlines = []
            try:
                for item in (yf.Ticker(ticker).news or [])[:5]:
                    content = item.get("content", item)
                    title = content.get("title", item.get("title", ""))
                    if title:
                        headlines.append(title)
            except Exception:
                pass
            sentiment = 0.0
            if groq_client and headlines:
                try:
                    joined = "\n".join(f"- {h}" for h in headlines[:5])
                    resp = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": f"Return ONLY a float -1.0 to 1.0 for sentiment of these {ticker} headlines:\n{joined}"}],
                        model="llama-3.3-70b-versatile", max_tokens=10,
                    )
                    sentiment = float(np.clip(float(resp.choices[0].message.content.strip()), -1.0, 1.0))
                except Exception:
                    pass
            features[ticker] = {"fundamental_score": fund_score, "news_sentiment": sentiment, "sector_risk": get_sector_risk(sector)}
        except Exception:
            features[ticker] = {"fundamental_score": 0.5, "news_sentiment": 0.0, "sector_risk": 1.0}
    return features


def _rolling_rsi(prices, window=14):
    if len(prices) < window + 1:
        return 0.5
    deltas = np.diff(prices)
    gain = np.mean(np.maximum(deltas, 0.0)[-window:])
    loss = np.mean(np.maximum(-deltas, 0.0)[-window:])
    if loss < 1e-10:
        return 1.0
    return float(1.0 - 1.0 / (1.0 + gain / loss))


def _rolling_macd(prices):
    if len(prices) < 26:
        return 0.0
    s = pd.Series(prices)
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(np.clip((macd - signal).iloc[-1] / (np.std(prices[-26:]) + 1e-8), -1.0, 1.0))


def _bollinger_pct(prices, window=20):
    if len(prices) < window:
        return 0.5
    s = pd.Series(prices[-window:])
    return float(np.clip((prices[-1] - (s.mean() - 2 * s.std())) / (4 * s.std() + 1e-8), 0.0, 1.0))


def _cap_and_renormalize(weights, cap=MAX_SINGLE_WEIGHT):
    capped = weights.copy()
    for _ in range(len(capped)):
        excess = np.maximum(capped - cap, 0.0)
        if excess.sum() < 1e-9:
            break
        capped = np.minimum(capped, cap)
        pool = excess.sum()
        free = capped < cap
        if not free.any():
            capped[:] = 1.0 / len(capped)
            break
        fw = capped[free].sum()
        if fw <= 1e-9:
            capped[:] = 1.0 / len(capped)
            break
        capped[free] += pool * (capped[free] / fw)
    total = capped.sum()
    return capped / total if total > 0 else capped


def _sortino_ratio(returns_arr, rf_daily=RISK_FREE_RATE_ANN / 252):
    if len(returns_arr) < 5:
        return 0.0
    excess = returns_arr - rf_daily
    dd_std = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) + 1e-8
    return float(np.mean(excess) / dd_std * np.sqrt(252))


def _volatility_regime(returns_arr, window=20):
    if len(returns_arr) < window:
        return 0.5
    recent = np.std(returns_arr[-window:])
    hist = np.std(returns_arr)
    if hist < 1e-10:
        return 0.5
    return float(np.clip(recent / (2.0 * hist), 0.0, 1.0))


def _ticker_hash(tickers):
    key = "_".join(sorted(t.upper() for t in tickers))
    return hashlib.md5(key.encode()).hexdigest()[:12]


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, progress_file, offset_pct=0.0, scale_pct=1.0):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_file   = progress_file
        self.offset_pct      = offset_pct
        self.scale_pct       = scale_pct
        self.current_reward  = 0.0

    def _on_step(self):
        try:
            self.current_reward += float(np.ravel(self.locals.get("rewards", [0]))[0])
        except Exception:
            pass
        raw = min(self.num_timesteps / self.total_timesteps, 1.0)
        overall = self.offset_pct + raw * self.scale_pct
        if self.num_timesteps % 100 == 0:
            with open(self.progress_file, "w") as fh:
                json.dump({"progress": round(overall * 100, 1), "timestep": self.num_timesteps,
                           "total_timesteps": self.total_timesteps,
                           "reward": round(float(self.current_reward), 4), "status": "training"}, fh)
        return True


class PortfolioEnv(gym.Env):
    WINDOW = 30
    INITIAL_CAPITAL = 100_000.0

    def __init__(self, price_data, ticker_features):
        super().__init__()
        self.price_data   = price_data.copy()
        self.returns      = price_data.pct_change().fillna(0)
        self.prices_arr   = price_data.values
        self.n_assets     = len(price_data.columns)
        self.tickers      = list(price_data.columns)
        self.ticker_features = ticker_features
        self._mom5  = price_data.pct_change(5).fillna(0).values
        self._mom20 = price_data.pct_change(20).fillna(0).values
        self._fund_scores  = np.array([ticker_features.get(t, {}).get("fundamental_score", 0.5) for t in self.tickers], dtype=np.float32)
        self._sentiments   = np.array([ticker_features.get(t, {}).get("news_sentiment",    0.0) for t in self.tickers], dtype=np.float32)
        self._sector_risks = np.array([ticker_features.get(t, {}).get("sector_risk",       1.0) for t in self.tickers], dtype=np.float32)
        obs_size = self.WINDOW * self.n_assets + 7 * self.n_assets + self.n_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space      = spaces.Box(low=0.0,    high=1.0,    shape=(self.n_assets,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = self.WINDOW
        self.portfolio_value = self.INITIAL_CAPITAL
        self.peak_value      = self.INITIAL_CAPITAL
        self.weights         = np.ones(self.n_assets) / self.n_assets
        self.prev_weights    = self.weights.copy()
        self.portfolio_history = [self.INITIAL_CAPITAL]
        return self._get_obs(), {}

    def _get_obs(self):
        idx    = self.current_step
        recent = self.returns.iloc[idx - self.WINDOW : idx].values
        if recent.shape[0] < self.WINDOW:
            recent = np.vstack([np.zeros((self.WINDOW - recent.shape[0], self.n_assets)), recent])
        mom5  = self._mom5[idx]  if idx < len(self._mom5)  else np.zeros(self.n_assets)
        mom20 = self._mom20[idx] if idx < len(self._mom20) else np.zeros(self.n_assets)
        lb    = min(idx, 60)
        rsi_v = np.zeros(self.n_assets, dtype=np.float32)
        mac_v = np.zeros(self.n_assets, dtype=np.float32)
        bol_v = np.zeros(self.n_assets, dtype=np.float32)
        for i in range(self.n_assets):
            sl = self.prices_arr[max(0, idx - lb):idx + 1, i]
            rsi_v[i] = _rolling_rsi(sl)
            mac_v[i] = _rolling_macd(sl)
            bol_v[i] = _bollinger_pct(sl)
        regime = _volatility_regime(self.returns.values[:idx].mean(axis=1))
        return np.concatenate([recent.flatten(), mom5, mom20, rsi_v, mac_v, bol_v,
                                self._fund_scores, self._sentiments, self.weights,
                                [self.portfolio_value / self.INITIAL_CAPITAL, regime]]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 0, None)
        s = action.sum()
        raw = action / s if s > 0 else np.ones(self.n_assets) / self.n_assets
        self.weights = _cap_and_renormalize(raw, MAX_SINGLE_WEIGHT)
        if self.current_step >= len(self.returns):
            return self._get_obs(), 0.0, True, False, {}
        turnover = float(np.sum(np.abs(self.weights - self.prev_weights)))
        tc = TRANSACTION_COST * turnover
        self.prev_weights = self.weights.copy()
        daily_ret = self.returns.iloc[self.current_step].values
        
        # Realism: Liquidity Penalty (Slippage)
        # High weights in low-volume stocks cause high market impact
        # We assume 0.1% slippage for every 1% of ADV consumed by the position
        # (Assuming a $1M portfolio for simulation purposes)
        ticker_adv = np.zeros(self.n_assets)
        for i, t in enumerate(self.tickers):
             # Proxy ADV from recent volume if available
             ticker_adv[i] = self.ticker_features.get(t, {}).get("avg_volume_30d", 1e6)
        
        simulated_size = self.weights * self.portfolio_value
        liquidity_pressure = simulated_size / (ticker_adv + 1e-8)
        slippage_penalty = np.sum(liquidity_pressure * 0.005) # 0.5% penalty for high ADV usage
        
        port_ret  = float(np.dot(self.weights, daily_ret)) - tc - slippage_penalty
        self.portfolio_value *= 1 + port_ret
        self.portfolio_history.append(self.portfolio_value)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        # Multi-factor Reward Function for Robust Backtesting Success
        # Base: log return (stable for RL convergence)
        reward = np.log(1.0 + port_ret)
        
        # 1. Enhanced Risk-Adjusted Return (Sortino)
        if len(self.portfolio_history) >= 20:
            h = np.array(self.portfolio_history[-60:])
            returns_seq = np.diff(h) / (h[:-1] + 1e-8)
            reward += 0.05 * _sortino_ratio(returns_seq)
            
        # 2. Risk Management (Drawdown & Sector Penalty)
        # We penalize drawdown quadratically to discourage "volatility chasing"
        reward -= float(np.dot(self.weights, self._sector_risks)) * 0.03 * (drawdown ** 2)
        
        # 3. Diversification Incentive (Entropy)
        # Encourages a smoother weight distribution
        reward += 0.01 * (-float(np.sum(self.weights * np.log(self.weights + 1e-8))))
        
        # 4. Fundamental & Sentiment Alignment
        # Reward alignment with high fundamental scores and positive news sentiment
        reward += 0.005 * float(np.dot(self.weights, self._fund_scores - 0.5))
        reward += 0.005 * float(np.dot(self.weights, self._sentiments))
        
        # 5. Stability (Turnover Penalty)
        # High turnover eats profits via costs; penalizing it creates more "believable" backtests
        reward -= 0.05 * turnover
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self._get_obs(), float(reward), done, False, {}

    def render(self): pass


def _make_a2c(env, hyperparams=None):
    hp = hyperparams or {}
    return A2C("MlpPolicy", env, verbose=0, learning_rate=hp.get("learning_rate", 7e-4),
               n_steps=int(hp.get("n_steps", 5)), gamma=hp.get("gamma", 0.99),
               gae_lambda=hp.get("gae_lambda", 1.0), ent_coef=hp.get("ent_coef", 0.01),
               policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))


def _make_sac(env, n_assets, hyperparams=None):
    hp = hyperparams or {}
    return SAC("MlpPolicy", env, verbose=0, learning_rate=hp.get("learning_rate", 3e-4),
               buffer_size=100_000, batch_size=int(hp.get("batch_size", 256)),
               tau=hp.get("tau", 0.005), gamma=hp.get("gamma", 0.99),
               ent_coef=hp.get("init_ent_coef", "auto"), policy_kwargs=dict(net_arch=[256, 256]))


def _make_td3(env, n_assets, hyperparams=None):
    hp = hyperparams or {}
    noise = NormalActionNoise(mean=np.zeros(n_assets), sigma=hp.get("noise_sigma", 0.1) * np.ones(n_assets))
    return TD3("MlpPolicy", env, verbose=0, learning_rate=hp.get("learning_rate", 1e-3),
               buffer_size=100_000, batch_size=int(hp.get("batch_size", 256)),
               tau=hp.get("tau", 0.005), gamma=hp.get("gamma", 0.99),
               action_noise=noise, policy_kwargs=dict(net_arch=[256, 256]))


class RLPortfolioAgent:
    AGENT_WEIGHTS     = {"A2C": 0.35, "SAC": 0.30, "TD3": 0.35}
    HPO_MIN_TIMESTEPS = 10000
    MODEL_DIR         = os.path.join(os.path.dirname(__file__), "rl_models")
    PROGRESS_DIR      = os.path.join(os.path.dirname(__file__), "rl_progress")
    HPO_CACHE_DIR     = os.path.join(os.path.dirname(__file__), "rl_hpo_cache")

    def __init__(self):
        os.makedirs(self.MODEL_DIR,    exist_ok=True)
        os.makedirs(self.PROGRESS_DIR, exist_ok=True)
        os.makedirs(self.HPO_CACHE_DIR, exist_ok=True)

    def _hpo_cache_path(self, tickers):
        return os.path.join(self.HPO_CACHE_DIR, f"{_ticker_hash(tickers)}.json")

    def _load_hpo_cache(self, tickers):
        path = self._hpo_cache_path(tickers)
        if os.path.exists(path):
            try:
                with open(path) as fh: return json.load(fh)
            except Exception: pass
        return None

    def _save_hpo_cache(self, tickers, hpo_results):
        with open(self._hpo_cache_path(tickers), "w") as fh:
            json.dump(hpo_results, fh, indent=2)

    def _ticker_hash_for_response(self, tickers):
        return _ticker_hash(tickers)

    def _fetch_data(self, tickers, period="2y", target_date=None):
        frames = {}
        for t in tickers:
            try:
                if target_date:
                    from datetime import timedelta
                    period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
                    days = period_days.get(period, 730)
                    end_dt = pd.to_datetime(target_date)
                    start_dt = end_dt - timedelta(days=days)
                    hist = yf.download(t, start=start_dt.strftime("%Y-%m-%d"), end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, threads=False, auto_adjust=False)
                else:
                    hist = yf.Ticker(t).history(period=period)
                
                if not hist.empty:
                    # Droplevel if MultiIndex (yfinance download can return this)
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist = hist.droplevel(-1, axis=1)
                    hist.index = pd.to_datetime(hist.index.date)
                    frames[t] = hist["Close"]
            except Exception: continue
        if len(frames) < 2:
            raise ValueError("Need at least 2 tickers with valid data.")
        df = pd.DataFrame(frames).ffill().dropna()
        if len(df) < 60:
            raise ValueError("Not enough aligned data.")
        return df

    def _write_progress(self, progress_file, progress, status, timestep=0, total_timesteps=0, reward=0, extra=None):
        data = {"progress": round(progress, 1), "status": status,
                "timestep": timestep, "total_timesteps": total_timesteps,
                "reward": round(float(reward), 4)}
        if extra: data.update(extra)
        with open(progress_file, "w") as fh: json.dump(data, fh)

    def train(self, tickers, timesteps=10000, session_id="default", use_hpo=True, target_date=None):
        if not GYM_AVAILABLE or not SB3_AVAILABLE:
            raise ImportError("Install: pip install stable-baselines3 gymnasium")
        progress_file = os.path.join(self.PROGRESS_DIR, f"{session_id}.json")
        self._write_progress(progress_file, 0, "fetching_data", total_timesteps=timesteps)
        price_data = self._fetch_data(tickers, target_date=target_date)
        self._write_progress(progress_file, 0, "fetching_fundamentals", total_timesteps=timesteps)
        ticker_features = fetch_ticker_features(tickers)

        evolved_hyperparams = None
        if use_hpo and timesteps >= self.HPO_MIN_TIMESTEPS:
            cached_hpo = self._load_hpo_cache(tickers)
            if cached_hpo:
                evolved_hyperparams = cached_hpo
                self._write_progress(progress_file, 5, "hpo_loaded_from_cache", total_timesteps=timesteps)
            else:
                self._write_progress(progress_file, 2, "hpo_running", total_timesteps=timesteps)
                try:
                    from .evolution import run_rl_hpo_all_agents
                    hpo_results = run_rl_hpo_all_agents(price_data, ticker_features, pop_size=3, generations=2, eval_timesteps=500)
                    evolved_hyperparams = {a: r["hyperparams"] for a, r in hpo_results.items() if r.get("hyperparams") is not None}
                    self._save_hpo_cache(tickers, evolved_hyperparams)
                    self._write_progress(progress_file, 20, "hpo_complete", total_timesteps=timesteps,
                                         extra={"hpo_source": "evolved", "evolved_hyperparams": evolved_hyperparams})
                except Exception as e:
                    evolved_hyperparams = None
                    self._write_progress(progress_file, 20, "hpo_skipped", total_timesteps=timesteps, extra={"hpo_error": str(e)})

        hpo_offset  = 20.0 if evolved_hyperparams is not None else 0.0
        agent_share = (99.0 - hpo_offset) / 3.0 / 100.0
        trained_models = {}
        all_rewards    = {}

        for agent_name, rel_offset in [("A2C", 0), ("SAC", agent_share), ("TD3", 2 * agent_share)]:
            abs_offset = (hpo_offset + rel_offset * 100) / 100.0
            self._write_progress(progress_file, hpo_offset + rel_offset * 100, f"training_{agent_name}",
                                  total_timesteps=timesteps, extra={"current_agent": agent_name})
            hp  = (evolved_hyperparams or {}).get(agent_name)
            vec = DummyVecEnv([lambda: PortfolioEnv(price_data, ticker_features)])
            cb  = ProgressCallback(timesteps, progress_file, offset_pct=abs_offset, scale_pct=agent_share)
            if agent_name == "A2C":
                model = _make_a2c(vec, hyperparams=hp)
            elif agent_name == "SAC":
                model = _make_sac(PortfolioEnv(price_data, ticker_features), len(tickers), hyperparams=hp)
            else:
                model = _make_td3(PortfolioEnv(price_data, ticker_features), len(tickers), hyperparams=hp)
            model.learn(total_timesteps=timesteps, callback=cb)
            model.save(os.path.join(self.MODEL_DIR, f"{session_id}_{agent_name}"))
            trained_models[agent_name] = model
            all_rewards[agent_name]    = float(cb.current_reward)

        self._write_progress(progress_file, 99, "evaluating", total_timesteps=timesteps)
        agent_allocs   = {}
        agent_histories = {}
        agent_dates    = {}
        for agent_name, model in trained_models.items():
            alloc, hist, dates = self._evaluate_single(model, price_data, ticker_features)
            agent_allocs[agent_name]    = alloc
            agent_histories[agent_name] = hist
            agent_dates[agent_name]     = dates

        ensemble_weights = np.zeros(len(tickers))
        for agent_name, w in self.AGENT_WEIGHTS.items():
            if agent_name in agent_allocs:
                arr = np.array([agent_allocs[agent_name].get(t, 0.0) for t in tickers])
                ensemble_weights += w * arr
        ensemble_weights = _cap_and_renormalize(ensemble_weights / (ensemble_weights.sum() + 1e-8))
        ensemble_alloc   = {t: round(float(w), 4) for t, w in zip(tickers, ensemble_weights)}

        primary_hist  = agent_histories["A2C"]
        final_value   = primary_hist[-1] if primary_hist else 100000
        total_return  = ((final_value / 100000) - 1) * 100

        agent_metrics = {}
        for agent_name, hist in agent_histories.items():
            if len(hist) > 1:
                rets = np.diff(hist) / (np.array(hist[:-1]) + 1e-8)
                agent_metrics[agent_name] = {
                    "final_value":  round(hist[-1], 2),
                    "total_return": round((hist[-1] / 100000 - 1) * 100, 2),
                    "sharpe":       round(float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)), 3),
                    "sortino":      round(_sortino_ratio(rets), 3),
                    "max_drawdown": round(float(self._max_drawdown(hist)), 4),
                    "allocation":   agent_allocs[agent_name],
                }

        result = {
            "progress": 100, "status": "complete",
            "timestep": timesteps, "total_timesteps": timesteps,
            "reward":   float(np.mean(list(all_rewards.values()))),
            "allocation": ensemble_alloc,
            "portfolio_history": primary_hist,
            "portfolio_dates":   agent_dates["A2C"],
            "tickers": tickers,
            "ticker_features": {t: {k: round(v, 3) for k, v in fm.items()} for t, fm in ticker_features.items()},
            "agent_metrics": agent_metrics,
            "final_value": final_value,
            "total_return": total_return,
            "hpo_used": evolved_hyperparams is not None,
            "hpo_source": "evolved" if evolved_hyperparams is not None else "defaults",
            "evolved_hyperparams": evolved_hyperparams or {},
        }
        with open(progress_file, "w") as fh: json.dump(result, fh)
        return result

    def _evaluate_single(self, model, price_data, ticker_features):
        env = PortfolioEnv(price_data, ticker_features)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        allocation = {t: round(float(w), 4) for t, w in zip(price_data.columns, env.weights)}
        history    = [round(v, 2) for v in env.portfolio_history]
        start_idx  = env.WINDOW
        date_index = price_data.index[start_idx:start_idx + len(history)]
        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in date_index]
        while len(dates) < len(history): dates.append("")
        return allocation, history, dates

    def _max_drawdown(self, history):
        arr  = np.array(history)
        peak = np.maximum.accumulate(arr)
        return float(((peak - arr) / (peak + 1e-8)).max())

    def get_progress(self, session_id):
        pf = os.path.join(self.PROGRESS_DIR, f"{session_id}.json")
        if not os.path.exists(pf):
            return {"progress": 0, "status": "not_started"}
        with open(pf) as fh: return json.load(fh)


_rl_agent = RLPortfolioAgent()
# trigger reload
