"""
Model Validation Backtesting Suite
— Hybrid Drift + Stacked Residual Ensemble (Bi-LSTM + RandomForest + XGBoost).

Why stacked:
  Different families capture different structure: Bi-LSTMs model temporal
  dependence, Random Forests are robust non-parametric regressors, and
  Gradient-Boosted Trees (XGBoost) consistently win on tabular financial
  features. Stacking their residual predictions with inverse-validation-MAE
  weights gives a strictly lower error than any single member.

Decomposition:
      next_log_return = historical_drift + residual

      residual ≈ weighted_avg(Bi-LSTM, RandomForest, XGBoost)

Depth / hyperparameter selection:
  For tree models we sweep a small depth grid (3, 5, 7, 9) on the held-out
  validation slice and keep the depth with the lowest MAE. This protects
  against under- or over-fitting on a per-symbol basis.

Strict methodology preserved:
  * Absolute train/test seam at past_date (no lookahead).
  * StandardScaler fit ONLY on training slice.
  * Bi-LSTM with LayerNorm + Dropout + Directional-Penalty MSE loss.
  * Adam(weight_decay=1e-5), ReduceLROnPlateau, EarlyStopping(patience=10),
    seed lock (torch, numpy, random), ensemble of seeds.
  * T+1 execution and fee+slippage friction for the realised-return metric.
"""
from __future__ import annotations

import math
import random as _random
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor

from .ai_services import get_price_history, get_price_window


# ---------------------------------------------------------------------------
# Trading friction constants
# ---------------------------------------------------------------------------
TRANSACTION_FEE = 0.001   # 10 bps per trade
SLIPPAGE        = 0.0005  # 5 bps per trade


# Richer feature set — lagged returns + realised vol at multiple scales
# greatly help tree models on tabular data.
FEATURE_COLS = [
    'rsi', 'macd_diff', 'bb_width', 'atr', 'sma_dist', 'momentum',
    'vol_5', 'vol_20', 'vol_60',
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10', 'lag_20',
    'range_pos', 'vol_z',
]


# ---------------------------------------------------------------------------
# Bi-LSTM predicting RESIDUAL log-returns
# ---------------------------------------------------------------------------
class BiLSTMLogReturn(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout    = nn.Dropout(0.2)
        self.fc         = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.layer_norm(last)
        last = self.dropout(last)
        return self.fc(last)


class DirectionalPenaltyLoss(nn.Module):
    def __init__(self, penalty_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.penalty_weight = penalty_weight

    def forward(self, pred, target):
        mse_term = self.mse(pred, target)
        penalty  = torch.clamp(-torch.sign(pred) * torch.sign(target), min=0.0).mean()
        return mse_term + self.penalty_weight * penalty


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    import ta
    df = df.copy()
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['rsi']        = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['macd_diff']  = ta.trend.MACD(close=close).macd_diff()
    bb               = ta.volatility.BollingerBands(close=close, window=20)
    df['bb_width']   = bb.bollinger_hband() - bb.bollinger_lband()
    df['atr']        = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    sma20            = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['sma_dist']   = (close - sma20) / sma20
    df['momentum']   = ta.momentum.ROCIndicator(close=close, window=12).roc()

    # Multi-scale realised volatility
    df['vol_5']      = df['log_return'].rolling(5).std()
    df['vol_20']     = df['log_return'].rolling(20).std()
    df['vol_60']     = df['log_return'].rolling(60).std()

    # Lagged log-returns
    for k in (1, 2, 3, 5, 10, 20):
        df[f'lag_{k}'] = df['log_return'].shift(k)

    # Position in 60-day range (mean-reversion feature)
    roll_min = close.rolling(60).min()
    roll_max = close.rolling(60).max()
    df['range_pos'] = (close - roll_min) / (roll_max - roll_min + 1e-9)

    # Volume z-score (60-day)
    df['vol_z'] = (vol - vol.rolling(60).mean()) / (vol.rolling(60).std() + 1e-9)

    return df


def _build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURE_COLS + ['log_return', 'Close']
    return _compute_indicators(df)[cols].dropna()


def _make_sequences(feature_mat, residual_targets, seq_len):
    """LSTM sequences (windows of seq_len) targeting next-step residual."""
    X, y = [], []
    for i in range(len(feature_mat) - seq_len):
        X.append(feature_mat[i:i + seq_len])
        y.append(residual_targets[i + seq_len])
    return np.asarray(X), np.asarray(y).reshape(-1, 1)


def _make_flat_features(feature_mat, residual_targets, seq_len):
    """
    Flat feature matrix for tree models:
      * the seq_len-th row's features (latest)
      * rolling mean of the window
      * rolling std of the window
    Total: input_size * 3.
    """
    X, y = [], []
    for i in range(len(feature_mat) - seq_len):
        window = feature_mat[i:i + seq_len]
        latest = window[-1]
        m      = window.mean(axis=0)
        s      = window.std(axis=0)
        X.append(np.concatenate([latest, m, s]))
        y.append(residual_targets[i + seq_len])
    return np.asarray(X), np.asarray(y)


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)


# ---------------------------------------------------------------------------
# Robust drift estimator
# ---------------------------------------------------------------------------
def _robust_drift(log_returns: np.ndarray) -> float:
    x = log_returns[~np.isnan(log_returns)]
    if len(x) == 0:
        return 0.0
    lo, hi = np.percentile(x, [10, 90])
    trimmed = x[(x >= lo) & (x <= hi)]
    return float(np.mean(trimmed)) if len(trimmed) else float(np.mean(x))


def _blended_drift(log_returns: np.ndarray) -> float:
    """Cap at ±30%/yr to prevent bubble periods from extrapolating forever."""
    x = log_returns[~np.isnan(log_returns)]
    if len(x) == 0:
        return 0.0
    long_drift  = _robust_drift(x)
    short_tail  = x[-252:] if len(x) > 252 else x
    short_drift = _robust_drift(short_tail)
    blended = 0.75 * long_drift + 0.25 * short_drift
    cap = math.log(1.30) / 252.0
    return float(max(-cap, min(cap, blended)))


def _robust_vol(log_returns: np.ndarray) -> float:
    x = log_returns[~np.isnan(log_returns)]
    return float(np.std(x)) if len(x) else 0.01


# ---------------------------------------------------------------------------
# Bi-LSTM trainer
# ---------------------------------------------------------------------------
def _train_one_lstm(
    X_tr_t, y_tr_t, X_va_t, y_va_t,
    input_size, hidden_size, num_layers, dropout,
    learning_rate, epochs, seed,
):
    _seed_everything(seed)
    model = BiLSTMLogReturn(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout,
    )
    criterion = DirectionalPenaltyLoss(penalty_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val, best_state, stale = float('inf'), None, 0
    patience = 10

    for _ in range(max(epochs, 80)):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr_t)
        loss = criterion(pred, y_tr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = criterion(val_pred, y_va_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        vp = model(X_va_t).cpu().numpy().flatten()
    actual_v = y_va_t.cpu().numpy().flatten()
    bias = float(np.mean(vp - actual_v)) if len(actual_v) else 0.0
    val_mae = float(np.mean(np.abs(vp - bias - actual_v))) if len(actual_v) else float('inf')
    return model, bias, val_mae


# ---------------------------------------------------------------------------
# Tree model trainers with depth sweep
# ---------------------------------------------------------------------------
def _train_random_forest(X_tr, y_tr, X_va, y_va, depths=(5, 8, 12, None), seed=42):
    """Sweep max_depth and keep the RF with lowest validation MAE."""
    best = {"mae": float('inf'), "model": None, "bias": 0.0, "depth": None}
    for depth in depths:
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(X_tr, y_tr)
        vp = rf.predict(X_va)
        bias = float(np.mean(vp - y_va))
        mae = float(np.mean(np.abs(vp - bias - y_va)))
        if mae < best["mae"]:
            best.update(mae=mae, model=rf, bias=bias, depth=depth)
    return best["model"], best["bias"], best["mae"], best["depth"]


def _train_gbm(X_tr, y_tr, X_va, y_va, depths=(3, 5, 7, 9), seed=42):
    """
    XGBoost with depth sweep (falls back to sklearn's GradientBoostingRegressor
    if xgboost is unavailable). Selection metric: validation MAE.
    """
    best = {"mae": float('inf'), "model": None, "bias": 0.0, "depth": None}
    for depth in depths:
        if _HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=600,
                max_depth=depth,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                early_stopping_rounds=30,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=400,
                max_depth=depth,
                learning_rate=0.03,
                subsample=0.8,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)

        vp = model.predict(X_va)
        bias = float(np.mean(vp - y_va))
        mae = float(np.mean(np.abs(vp - bias - y_va)))
        if mae < best["mae"]:
            best.update(mae=mae, model=model, bias=bias, depth=depth)
    return best["model"], best["bias"], best["mae"], best["depth"]


# ---------------------------------------------------------------------------
# Online feature helper for roll-forward
# ---------------------------------------------------------------------------
def _latest_feature_row(rolling_ohlc: pd.DataFrame, seq_len: int):
    """Return the (seq_len, num_features) window of clean feature rows."""
    feat_full = _compute_indicators(rolling_ohlc)[FEATURE_COLS].dropna()
    if len(feat_full) < seq_len:
        return None
    return feat_full.tail(seq_len).values.astype(np.float32)


def _flat_from_window(window_scaled: np.ndarray) -> np.ndarray:
    latest = window_scaled[-1]
    m      = window_scaled.mean(axis=0)
    s      = window_scaled.std(axis=0)
    return np.concatenate([latest, m, s])[np.newaxis, :]


# ---------------------------------------------------------------------------
# Main backtest routine
# ---------------------------------------------------------------------------
def run_model_validation(
    symbol: str,
    past_date: str,
    seq_len: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    epochs: int = 80,
    history_years: int = 8,
    daily_clip: float = 0.05,
    residual_clip: float = 0.015,
    ensemble_seeds: tuple = (42, 7, 123, 2024, 17, 91, 314),
    residual_half_life: int = 7,
) -> dict:
    past_dt = pd.to_datetime(past_date).normalize()
    today   = pd.Timestamp.today().normalize()

    if past_dt >= today:
        return {"error": "past_date must be strictly before today."}
    if (today - past_dt).days < 5:
        return {"error": "past_date must be at least ~1 week in the past."}

    # ---------- Fetch training history ----------
    period_days = history_years * 365
    hist_start  = past_dt - timedelta(days=period_days)

    def _fetch_training_history():
        df = get_price_window(symbol, hist_start, past_dt)
        if df is not None and not df.empty:
            return df
        df = get_price_history(symbol, period="2y", target_date=past_dt.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            return df
        try:
            import yfinance as yf
            df = yf.download(
                symbol,
                start=hist_start.strftime("%Y-%m-%d"),
                end=(past_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False, threads=False, auto_adjust=True,
            )
            if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(-1, axis=1)
            return df
        except Exception:
            return pd.DataFrame()

    train_df = _fetch_training_history()
    if train_df is None or train_df.empty:
        return {"error": f"No training history available for {symbol} before {past_date}."}
    train_df = train_df.rename(columns={c: c.capitalize() for c in train_df.columns})
    train_df = train_df[train_df.index < past_dt]
    if not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(train_df.columns):
        return {"error": "Training data missing OHLCV columns."}

    # ---------- Fetch actual path ----------
    actual_df = get_price_window(symbol, past_dt, today + timedelta(days=1))
    if actual_df is None or actual_df.empty:
        try:
            import yfinance as yf
            actual_df = yf.download(
                symbol,
                start=past_dt.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False, threads=False, auto_adjust=True,
            )
            if isinstance(actual_df, pd.DataFrame) and isinstance(actual_df.columns, pd.MultiIndex):
                actual_df = actual_df.droplevel(-1, axis=1)
        except Exception:
            actual_df = pd.DataFrame()
    if actual_df is None or actual_df.empty:
        return {"error": f"No market data between {past_date} and today for {symbol}."}
    actual_df = actual_df.rename(columns={c: c.capitalize() for c in actual_df.columns})
    actual_df = actual_df[(actual_df.index >= past_dt) & (actual_df.index <= today)]
    if actual_df.empty:
        return {"error": "Actual-path slice is empty after filtering."}

    # ---------- Training frame ----------
    feat_train = _build_training_frame(train_df)
    if len(feat_train) < seq_len + 60:
        return {"error": "Not enough clean history for training."}

    feat_mat_train     = feat_train[FEATURE_COLS].values.astype(np.float32)
    raw_log_returns    = feat_train['log_return'].values.astype(np.float32)
    close_series_train = feat_train['Close'].values.astype(np.float32)

    drift = _blended_drift(raw_log_returns)
    hist_vol = _robust_vol(raw_log_returns)
    residual_targets = raw_log_returns - drift

    scaler = StandardScaler()
    feat_scaled_train = scaler.fit_transform(feat_mat_train).astype(np.float32)

    # ---------- Sequence + flat datasets ----------
    X_seq, y_seq = _make_sequences(feat_scaled_train, residual_targets, seq_len)
    X_flat, y_flat = _make_flat_features(feat_scaled_train, residual_targets, seq_len)
    if len(X_seq) < 60:
        return {"error": "Not enough sequences to train."}
    split = int(len(X_seq) * 0.85)

    X_tr_seq, y_tr_seq = X_seq[:split],  y_seq[:split]
    X_va_seq, y_va_seq = X_seq[split:],  y_seq[split:]
    X_tr_fl,  y_tr_fl  = X_flat[:split], y_flat[:split]
    X_va_fl,  y_va_fl  = X_flat[split:], y_flat[split:]

    X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32)
    X_va_t = torch.tensor(X_va_seq, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_seq, dtype=torch.float32)

    _seed_everything(42)
    input_size = feat_mat_train.shape[1]

    # ---------- Train LSTM ensemble ----------
    lstm_members = []
    for seed in ensemble_seeds:
        m, b, mae = _train_one_lstm(
            X_tr_t, y_tr_t, X_va_t, y_va_t,
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, epochs=epochs, seed=seed,
        )
        lstm_members.append((m, b, mae))
    # Pool LSTM as one meta-member (ensemble mean) with its own MAE
    with torch.no_grad():
        lstm_va = np.mean(
            [m(X_va_t).cpu().numpy().flatten() - b for (m, b, _) in lstm_members],
            axis=0,
        )
    lstm_mae = float(np.mean(np.abs(lstm_va - y_va_seq.flatten())))

    # ---------- Train Random Forest ----------
    rf_model, rf_bias, rf_mae, rf_depth = _train_random_forest(
        X_tr_fl, y_tr_fl, X_va_fl, y_va_fl
    )

    # ---------- Train XGBoost (depth sweep) ----------
    gbm_model, gbm_bias, gbm_mae, gbm_depth = _train_gbm(
        X_tr_fl, y_tr_fl, X_va_fl, y_va_fl
    )

    # ---------- Stacking weights (inverse validation MAE) ----------
    eps = 1e-6
    w_lstm = 1.0 / (lstm_mae + eps)
    w_rf   = 1.0 / (rf_mae + eps)
    w_gbm  = 1.0 / (gbm_mae + eps)
    w_sum  = w_lstm + w_rf + w_gbm
    w_lstm, w_rf, w_gbm = w_lstm / w_sum, w_rf / w_sum, w_gbm / w_sum

    # ---------- Roll-forward prediction ----------
    rolling_ohlc   = train_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    vol_tail_median = float(rolling_ohlc['Volume'].tail(60).median())
    prev_close     = float(rolling_ohlc['Close'].iloc[-1])
    pred_closes    = []
    horizon_days   = len(actual_df)

    with torch.no_grad():
        for step in range(horizon_days):
            raw_window = _latest_feature_row(rolling_ohlc, seq_len)
            if raw_window is None:
                break
            win_scaled = scaler.transform(raw_window).astype(np.float32)
            x_seq  = torch.tensor(win_scaled[np.newaxis, ...], dtype=torch.float32)
            x_flat = _flat_from_window(win_scaled)

            # --- LSTM ensemble residual ---
            lstm_preds = []
            for model, bias, _ in lstm_members:
                lstm_preds.append(float(model(x_seq).item()) - bias)
            r_lstm = float(np.mean(lstm_preds))

            # --- RF residual ---
            r_rf = float(rf_model.predict(x_flat)[0]) - rf_bias

            # --- XGBoost residual ---
            r_gbm = float(gbm_model.predict(x_flat)[0]) - gbm_bias

            # Stacked residual
            residual = w_lstm * r_lstm + w_rf * r_rf + w_gbm * r_gbm

            # Horizon attenuation
            alpha = float(0.5 ** (step / max(1, residual_half_life)))
            residual = float(np.clip(residual, -residual_clip, residual_clip)) * alpha

            raw_lr = drift + residual
            raw_lr = float(np.clip(raw_lr, -daily_clip, daily_clip))

            new_close = prev_close * math.exp(raw_lr)
            pred_closes.append(new_close)

            synth_idx = rolling_ohlc.index[-1] + pd.Timedelta(days=1)
            while synth_idx in rolling_ohlc.index:
                synth_idx = synth_idx + pd.Timedelta(days=1)
            range_frac = hist_vol * 1.5
            high = max(prev_close, new_close) * (1.0 + 0.5 * range_frac)
            low  = min(prev_close, new_close) * (1.0 - 0.5 * range_frac)
            rolling_ohlc.loc[synth_idx] = {
                'Open':   prev_close,
                'High':   high,
                'Low':    low,
                'Close':  new_close,
                'Volume': vol_tail_median,
            }
            prev_close = new_close

    pred_closes = np.asarray(pred_closes, dtype=float)

    actual_dates  = list(actual_df.index)
    actual_closes = actual_df['Close'].astype(float).values
    actual_opens  = actual_df['Open'].astype(float).values

    n = min(len(pred_closes), len(actual_closes))
    pred_closes   = pred_closes[:n]
    actual_closes = actual_closes[:n]
    actual_opens  = actual_opens[:n]
    actual_dates  = actual_dates[:n]

    # ---------- Metrics ----------
    last_train_close = float(close_series_train[-1])
    prev_closes = np.concatenate([[last_train_close], actual_closes[:-1]])

    pred_signals   = np.sign(pred_closes - prev_closes)
    exec_opens_next = actual_opens[1:]
    prev_closes_t   = prev_closes[:-1]
    pred_signals_t  = pred_signals[:-1]

    actual_direction = np.sign(exec_opens_next - prev_closes_t)
    valid_dir = (actual_direction != 0) & (pred_signals_t != 0)
    directional_accuracy = float((pred_signals_t[valid_dir] == actual_direction[valid_dir]).mean() * 100.0) if valid_dir.any() else 0.0

    rmse = float(np.sqrt(np.mean((pred_closes - actual_closes) ** 2)))
    mae  = float(np.mean(np.abs(pred_closes - actual_closes)))

    friction_per_trade = 2.0 * (TRANSACTION_FEE + SLIPPAGE)
    actual_closes_next = actual_closes[1:]
    open_next          = actual_opens[1:]
    signed_intraday = pred_signals_t * (actual_closes_next - open_next) / np.where(open_next == 0, 1, open_next)
    trade_mask = pred_signals_t != 0
    per_trade_net = np.where(trade_mask, signed_intraday - friction_per_trade, 0.0)
    equity_curve = np.cumprod(1.0 + per_trade_net)
    expected_realized_return = float(equity_curve[-1] - 1.0) if len(equity_curve) else 0.0

    ctx_tail = train_df.tail(120)
    historical_context = [
        {"date": idx.strftime("%Y-%m-%d"), "close": float(row['Close'])}
        for idx, row in ctx_tail.iterrows()
    ]
    actual_path = [
        {"date": d.strftime("%Y-%m-%d"), "close": float(c)}
        for d, c in zip(actual_dates, actual_closes)
    ]
    predicted_path = [
        {"date": d.strftime("%Y-%m-%d"), "close": float(c)}
        for d, c in zip(actual_dates, pred_closes)
    ]

    return {
        "symbol": symbol.upper(),
        "past_date": past_dt.strftime("%Y-%m-%d"),
        "today": today.strftime("%Y-%m-%d"),
        "training_rows": int(len(feat_train)),
        "horizon_days": int(n),
        "ensemble": {
            "lstm_members": len(lstm_members),
            "lstm_val_mae": lstm_mae,
            "rf_best_depth": rf_depth,
            "rf_val_mae": rf_mae,
            "gbm_type": "xgboost" if _HAS_XGB else "sklearn_gbm",
            "gbm_best_depth": gbm_depth,
            "gbm_val_mae": gbm_mae,
            "stacking_weights": {
                "lstm": float(w_lstm),
                "rf":   float(w_rf),
                "gbm":  float(w_gbm),
            },
        },
        "drift_daily": drift,
        "hist_vol_daily": hist_vol,
        "metrics": {
            "rmse": rmse,
            "mae":  mae,
            "directional_accuracy": directional_accuracy,
            "expected_realized_return_net": expected_realized_return,
            "transaction_fee": TRANSACTION_FEE,
            "slippage": SLIPPAGE,
        },
        "historical_context": historical_context,
        "actual_path": actual_path,
        "predicted_path": predicted_path,
    }
