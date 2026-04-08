import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from unittest.mock import patch

from .ai_services import PortfolioOptimizer, TrendPredictor
from .bl_numpy import (
    bl_posterior,
    build_omega_idzorek,
    compute_asset_betas,
    select_market_benchmark,
)
from .tasks import queue_ga_for_ticker


class BlackLittermanOmegaTests(SimpleTestCase):
    def test_confidence_calibration_moves_posterior_toward_view(self):
        Sigma = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
        Pi = np.array([[0.06], [0.08]], dtype=float)
        P = np.array([[1.0, 0.0]])
        Q = np.array([[0.12]])
        market_weights = np.array([0.6, 0.4])

        Omega, _ = build_omega_idzorek(P, Q, 0.05, Sigma, Pi, market_weights, [75])
        posterior_returns, _ = bl_posterior(0.05, Sigma, Pi, P, Q, Omega)

        self.assertGreater(posterior_returns[0], Pi.flatten()[0] + 0.01)


class BlackLittermanAnalyticsTests(SimpleTestCase):
    def test_select_market_benchmark_uses_sp500_for_us_tickers(self):
        self.assertEqual(select_market_benchmark(["AAPL", "MSFT", "NVDA"]), "^GSPC")

    def test_select_market_benchmark_uses_nifty_for_indian_tickers(self):
        self.assertEqual(select_market_benchmark(["RELIANCE.NS", "TCS.NS", "INFY.NS"]), "^NSEI")

    def test_compute_asset_betas_uses_shared_return_history(self):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        benchmark = pd.Series([100, 101, 102, 103, 104], index=dates, name="benchmark")
        prices = pd.DataFrame(
            {
                "AAPL": [100, 102, 104, 106, 108],
                "MSFT": [100, 101, 102, 103, 104],
            },
            index=dates,
        )

        betas = compute_asset_betas(prices, benchmark)
        self.assertGreater(betas["AAPL"], betas["MSFT"])
        self.assertGreater(betas["MSFT"], 0)


class PortfolioOptimizerBlendTests(SimpleTestCase):
    def test_blend_profiles_interpolates_smoothly(self):
        conservative = pd.Series({"AAPL": 0.7, "MSFT": 0.3})
        balanced = pd.Series({"AAPL": 0.4, "MSFT": 0.6})
        aggressive = pd.Series({"AAPL": 0.1, "MSFT": 0.9})

        low_risk, low_meta = PortfolioOptimizer._blend_profiles(
            conservative, balanced, aggressive, 0.25
        )
        high_risk, high_meta = PortfolioOptimizer._blend_profiles(
            conservative, balanced, aggressive, 0.75
        )

        self.assertAlmostEqual(low_risk["AAPL"], 0.55)
        self.assertAlmostEqual(high_risk["AAPL"], 0.25)
        self.assertEqual(low_meta["from"], "HRP")
        self.assertEqual(high_meta["to"], "Max Sharpe")


class PredictionFallbackTests(SimpleTestCase):
    @patch("api.ai_services.get_price_history")
    def test_forecast_price_uses_historical_mean_fallback_when_lstm_disabled(self, mock_get_price_history):
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        asset_close = pd.Series(100 * (1.0010 ** np.arange(len(dates))), index=dates)
        benchmark_close = pd.Series(100 * (1.0005 ** np.arange(len(dates))), index=dates)

        def fake_history(symbol, period="2y"):
            if symbol == "AAPL":
                return pd.DataFrame({"Close": asset_close})
            if symbol == "SPY":
                return pd.DataFrame({"Close": benchmark_close})
            return pd.DataFrame()

        mock_get_price_history.side_effect = fake_history

        forecast = TrendPredictor.forecast_price("AAPL", allow_lstm=False)

        self.assertIsNotNone(forecast)
        self.assertEqual(forecast["prediction_method"], "historical_mean_fallback")
        self.assertIn("Historical mean", forecast["prediction_label"])
        self.assertGreater(forecast["predicted_price"], float(asset_close.iloc[-1]))


class CeleryQueueHelperTests(SimpleTestCase):
    @patch("api.tasks.run_ga_for_ticker.delay")
    @patch("api.tasks.celery_broker_available", return_value=False)
    def test_queue_ga_for_ticker_skips_when_broker_unavailable(self, _mock_available, mock_delay):
        queued = queue_ga_for_ticker("AAPL")
        self.assertFalse(queued)
        mock_delay.assert_not_called()

    @patch("api.tasks.run_ga_for_ticker.delay")
    @patch("api.tasks.celery_broker_available", return_value=True)
    def test_queue_ga_for_ticker_dispatches_when_broker_available(self, _mock_available, mock_delay):
        queued = queue_ga_for_ticker("AAPL")
        self.assertTrue(queued)
        mock_delay.assert_called_once()
