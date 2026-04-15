import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from pmdarima import auto_arima
from arch import arch_model

# internal
from .ai_services import get_price_history  # reuse existing data fetcher


class ClassicalForecaster:
    def __init__(self, symbol: str, past_date: str):
        self.symbol = symbol
        self.past_date = datetime.strptime(past_date, "%Y-%m-%d")
        self.arima_model = None
        self.garch_model = None
        self.scaler_mean = None  # mean of log returns on train set (for centering)
        self.scaler_std = None   # std of log returns on train set (for centering)

    def _fetch_and_split(self):
        """Fetch 2y daily OHLCV, split at past_date, return train/test price series."""
        df = get_price_history(self.symbol, period="5y", target_date=None)
        if df is None or df.empty:
            raise ValueError(f"No price data for {self.symbol}")

        # yfinance sometimes returns a MultiIndex on columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        train_prices = df.loc[df.index <= self.past_date, "Close"]
        test_prices  = df.loc[df.index >  self.past_date, "Close"]

        if len(train_prices) < 30:
            raise ValueError("Insufficient training data (need 30+ days before past_date)")

        return train_prices, test_prices

    def _to_log_returns(self, prices: pd.Series) -> np.ndarray:
        return np.log(prices / prices.shift(1)).dropna().values

    def fit(self, train_prices: pd.Series):
        """Fit ARIMA on log returns, then fit GARCH(1,1) on ARIMA residuals."""
        log_ret = self._to_log_returns(train_prices)

        # center log returns (helps GARCH stability)
        self.scaler_mean = log_ret.mean()
        self.scaler_std  = log_ret.std()
        log_ret_scaled = (log_ret - self.scaler_mean) / self.scaler_std

        # auto_arima: AIC criterion, seasonal=False (daily returns have no seasonality),
        # stepwise search for speed, max_p=5, max_q=3, d determined by ADF test
        self.arima_model = auto_arima(
            log_ret_scaled,
            start_p=1, start_q=1,
            max_p=5, max_q=3,
            max_d=2,
            d=None,           # auto-determine via ADF test
            test="adf",
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

        arima_residuals = self.arima_model.resid()

        # GARCH(1,1) on residuals — industry standard, no need for higher orders
        garch_spec = arch_model(arima_residuals, vol="Garch", p=1, q=1, dist="normal")
        self.garch_result = garch_spec.fit(disp="off")

    def forecast(self, n_steps: int, last_train_price: float) -> dict:
        """
        Produce n_steps ahead point forecast + 95% confidence interval.
        Returns prices (not log returns) by inverting the log return cumulation.
        """
        # ARIMA point forecast (scaled log returns space)
        arima_preds_scaled, conf_int = self.arima_model.predict(
            n_periods=n_steps, return_conf_int=True, alpha=0.05
        )

        # unscale back to raw log return space
        arima_preds = arima_preds_scaled * self.scaler_std + self.scaler_mean
        lower_raw   = conf_int[:, 0] * self.scaler_std + self.scaler_mean
        upper_raw   = conf_int[:, 1] * self.scaler_std + self.scaler_mean

        # GARCH variance forecast to widen confidence bands
        garch_forecast = self.garch_result.forecast(horizon=min(n_steps, 30))
        garch_var = garch_forecast.variance.values[-1]  # shape (horizon,)
        # if n_steps > 30, pad last garch variance
        if n_steps > len(garch_var):
            garch_var = np.concatenate([
                garch_var,
                np.full(n_steps - len(garch_var), garch_var[-1])
            ])
        garch_std = np.sqrt(garch_var[:n_steps]) * self.scaler_std

        # widen confidence bands using GARCH conditional std
        lower_raw = lower_raw - 1.96 * garch_std
        upper_raw = upper_raw + 1.96 * garch_std

        # cumulative sum of log returns → price levels
        # price[t] = last_train_price * exp(sum of log returns up to t)
        cum_log_ret       = np.cumsum(arima_preds)
        cum_log_ret_lower = np.cumsum(lower_raw)
        cum_log_ret_upper = np.cumsum(upper_raw)

        price_forecast = last_train_price * np.exp(cum_log_ret)
        price_lower    = last_train_price * np.exp(cum_log_ret_lower)
        price_upper    = last_train_price * np.exp(cum_log_ret_upper)

        max_deviation = 0.5  # cap bands at ±50% of last known price
        price_lower = np.clip(price_lower, last_train_price * (1 - max_deviation),
                              last_train_price * (1 + max_deviation))
        price_upper = np.clip(price_upper, last_train_price * (1 - max_deviation),
                              last_train_price * (1 + max_deviation))

        return {
            "predicted": price_forecast.tolist(),
            "lower":     price_lower.tolist(),
            "upper":     price_upper.tolist(),
        }

    def run_classical_backtest(self) -> dict:
        """
        Full pipeline. Returns dict matching the shape expected by the frontend,
        plus confidence interval arrays.
        """
        train_prices, test_prices = self._fetch_and_split()

        if test_prices.empty:
            raise ValueError("No test data found after past_date")

        self.fit(train_prices)

        n_steps = len(test_prices)
        last_train_price = float(train_prices.iloc[-1])
        forecast_result = self.forecast(n_steps, last_train_price)

        pred_prices  = np.array(forecast_result["predicted"])
        lower_prices = np.array(forecast_result["lower"])
        upper_prices = np.array(forecast_result["upper"])
        actual_prices = test_prices.values
        test_dates    = [d.strftime("%Y-%m-%d") for d in test_prices.index]
        hist_dates    = [d.strftime("%Y-%m-%d") for d in train_prices.index]
        hist_prices   = train_prices.values.tolist()

        # metrics
        rmse = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))
        mae  = float(np.mean(np.abs(pred_prices - actual_prices)))

        actual_log_ret = np.log(actual_prices[1:] / actual_prices[:-1])
        pred_log_ret   = np.log(pred_prices[1:] / pred_prices[:-1])
        direction_correct = np.sign(actual_log_ret) == np.sign(pred_log_ret)
        directional_accuracy = float(direction_correct.mean() * 100)

        # simple realized return: buy at test start, measure predicted vs actual
        expected_return = float((pred_prices[-1] / last_train_price - 1) * 100)
        actual_return   = float((actual_prices[-1] / last_train_price - 1) * 100)

        # arima order string for display
        arima_order = str(self.arima_model.order)

        # Per-row trade log
        actual_log_ret_signs = np.sign(np.log(actual_prices[1:] / actual_prices[:-1]))
        pred_log_ret_signs   = np.sign(np.log(pred_prices[1:]  / pred_prices[:-1]))

        trade_log = []
        for k in range(len(test_dates)):
            trade_log.append({
                "date":              test_dates[k],
                "predicted_price":   round(float(pred_prices[k]), 2),
                "actual_price":      round(float(actual_prices[k]), 2),
                "error_abs":         round(float(abs(pred_prices[k] - actual_prices[k])), 2),
                "error_pct":         round(float(abs(pred_prices[k] - actual_prices[k]) / actual_prices[k] * 100), 2),
                "direction_correct": bool(
                    k == 0 or actual_log_ret_signs[k-1] == pred_log_ret_signs[k-1]
                ),
            })

        # Win rate
        win_rate = float(
            sum(r["direction_correct"] for r in trade_log) / len(trade_log) * 100
        ) if trade_log else 0.0

        # Equity curves
        TRANSACTION_FEE = 0.001
        SLIPPAGE        = 0.0005

        arima_equity   = [100.0]
        buyhold_equity = [100.0]

        for k in range(len(actual_prices) - 1):
            signal = pred_log_ret_signs[k] if k < len(pred_log_ret_signs) else 0
            entry  = actual_prices[k]
            exit_p = actual_prices[k + 1]
            if entry > 0 and signal != 0:
                raw_ret = signal * (exit_p - entry) / entry
                net_ret = raw_ret - TRANSACTION_FEE - SLIPPAGE
            else:
                net_ret = 0.0
            arima_equity.append(arima_equity[-1] * (1 + net_ret))

            bh_ret = (exit_p - entry) / entry if entry > 0 else 0.0
            buyhold_equity.append(buyhold_equity[-1] * (1 + bh_ret))

        equity_dates = test_dates[:len(arima_equity)]

        # Sharpe
        arima_returns = np.diff(arima_equity) / np.array(arima_equity[:-1])
        if len(arima_returns) > 1 and arima_returns.std() > 0:
            sharpe = float((arima_returns.mean() / arima_returns.std()) * np.sqrt(52))
        else:
            sharpe = 0.0

        # Max drawdown
        peak = arima_equity[0]
        max_dd = 0.0
        for val in arima_equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        max_drawdown = float(max_dd * 100)

        return {
            "historical": {
                "dates":  hist_dates,
                "prices": hist_prices,
            },
            "test": {
                "dates":     test_dates,
                "actual":    actual_prices.tolist(),
                "predicted": pred_prices.tolist(),
                "lower":     lower_prices.tolist(),
                "upper":     upper_prices.tolist(),
            },
            "metrics": {
                "rmse":                    rmse,
                "mae":                     mae,
                "directional_accuracy":    directional_accuracy,
                "expected_realized_return": expected_return,
                "actual_realized_return":  actual_return,
                "arima_order":             arima_order,
                "win_rate":     round(win_rate, 1),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown": round(max_drawdown, 2),
            },
            "trade_log": trade_log,
            "equity_curve": {
                "dates":          equity_dates,
                "arima_equity":   [round(v, 4) for v in arima_equity],
                "buyhold_equity": [round(v, 4) for v in buyhold_equity],
            },
        }
