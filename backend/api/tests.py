import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
from django.test import SimpleTestCase
from unittest.mock import patch
from django.db.utils import OperationalError, ProgrammingError
from rest_framework.test import APIRequestFactory

from .ai_services import PortfolioOptimizer, TrendPredictor
from .bl_numpy import (
    bl_posterior,
    build_omega_idzorek,
    compute_asset_betas,
    run_bl_analysis,
    select_market_benchmark,
)
from .black_scholes import bs_call_price, bs_put_price, greeks, implied_volatility
from .models import SearchedTicker
from .tasks import queue_ga_for_ticker

FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from reporting_utils import resolve_report_universe


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

    @patch("api.ai_services.get_price_history")
    def test_optimize_changes_meaningfully_across_risk_levels(self, mock_get_price_history):
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        steps = np.arange(len(dates), dtype=float)
        histories = {
            "AAPL": 100 * np.cumprod(1 + 0.00055 + 0.00025 * np.sin(steps / 17)),
            "MSFT": 95 * np.cumprod(1 + 0.00045 + 0.00020 * np.cos(steps / 19)),
            "GOOG": 90 * np.cumprod(1 + 0.00100 + 0.00120 * np.sin(steps / 9)),
            "TSLA": 80 * np.cumprod(1 + 0.00390 + 0.00050 * np.sin(steps / 5)),
        }

        def fake_history(symbol, period="2y"):
            close = histories.get(symbol)
            if close is None:
                return pd.DataFrame()
            return pd.DataFrame({"Close": close}, index=dates)

        mock_get_price_history.side_effect = fake_history

        low = pd.Series(PortfolioOptimizer.optimize(["AAPL", "MSFT", "GOOG", "TSLA"], 0.0)["allocation"])
        mid = pd.Series(PortfolioOptimizer.optimize(["AAPL", "MSFT", "GOOG", "TSLA"], 0.5)["allocation"])
        high = pd.Series(PortfolioOptimizer.optimize(["AAPL", "MSFT", "GOOG", "TSLA"], 1.0)["allocation"])

        self.assertAlmostEqual(float(low.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(mid.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(high.sum()), 1.0, places=3)
        self.assertTrue(not low.equals(high))
        self.assertGreaterEqual(float((low - high).abs().max()), 0.05)
        self.assertGreaterEqual(float((low - high).abs().sum()), 0.20)


class StreamlitPageConfigTests(SimpleTestCase):
    def test_streamlit_page_filenames_are_unique_after_numeric_prefix(self):
        page_dir = FRONTEND_ROOT / "pages"
        inferred = {}

        for page in page_dir.glob("*.py"):
            inferred_name = re.sub(r"^\d+_", "", page.stem)
            inferred.setdefault(inferred_name, []).append(page.name)

        duplicates = {name: files for name, files in inferred.items() if len(files) > 1}
        self.assertEqual(duplicates, {})


class SearchedTickerResilienceTests(SimpleTestCase):
    @patch("api.models.SearchedTicker.objects.get_or_create", side_effect=OperationalError("schema mismatch"))
    def test_log_ignores_database_schema_errors(self, _mock_get_or_create):
        SearchedTicker.log("AAPL")

    @patch("api.models.SearchedTicker.objects.values_list", side_effect=ProgrammingError("schema mismatch"))
    def test_get_watchlist_returns_empty_list_on_schema_errors(self, _mock_values_list):
        self.assertEqual(SearchedTicker.get_watchlist(), [])


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


class ReportUniverseTests(SimpleTestCase):
    def test_resolve_report_universe_keeps_zero_weight_requested_assets(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        data_frame = pd.DataFrame(
            {
                "AAPL": np.linspace(100, 110, len(dates)),
                "MSFT": np.linspace(90, 95, len(dates)),
                "SPY": np.linspace(400, 410, len(dates)),
                "QQQ": np.linspace(300, 315, len(dates)),
            },
            index=dates,
        )
        resolved = resolve_report_universe(
            ["AAPL", "MSFT", "SPY", "QQQ"],
            {"SPY": 0.8589, "MSFT": 0.1411, "AAPL": 0.0, "QQQ": 0.0},
            data_frame,
            data_frame,
            data_frame,
        )

        self.assertEqual(resolved["analysis_tickers"], ["AAPL", "MSFT", "SPY", "QQQ"])
        self.assertIn("AAPL", resolved["allocation_weights"])
        self.assertEqual(resolved["allocation_weights"]["AAPL"], 0.0)
        self.assertEqual(resolved["allocation_weights"]["QQQ"], 0.0)


class BlackScholesEngineTests(SimpleTestCase):
    def test_put_call_parity_holds_with_dividend_yield(self):
        call_price = bs_call_price(100, 100, 0.5, 0.05, 0.2, q=0.02)
        put_price = bs_put_price(100, 100, 0.5, 0.05, 0.2, q=0.02)
        lhs = call_price - put_price
        rhs = 100 * np.exp(-0.02 * 0.5) - 100 * np.exp(-0.05 * 0.5)
        self.assertAlmostEqual(lhs, rhs, places=6)

    def test_dividend_yield_reduces_call_delta(self):
        no_dividend_delta = greeks(100, 100, 0.5, 0.05, 0.2, q=0.0)["call"]["delta"]
        with_dividend_delta = greeks(100, 100, 0.5, 0.05, 0.2, q=0.03)["call"]["delta"]
        self.assertLess(with_dividend_delta, no_dividend_delta)

    def test_implied_volatility_recovers_theoretical_sigma(self):
        market_price = bs_call_price(100, 100, 0.5, 0.05, 0.24, q=0.01)
        solved_sigma = implied_volatility(market_price, 100, 100, 0.5, 0.05, q=0.01, option_type="call")
        self.assertIsNotNone(solved_sigma)
        self.assertAlmostEqual(float(solved_sigma), 0.24, places=3)

    def test_options_page_uses_shared_black_scholes_renderer(self):
        options_page = Path(__file__).resolve().parents[2] / "frontend" / "pages" / "3_Options_Analysis.py"
        source = options_page.read_text(encoding="utf-8")
        self.assertIn("render_black_scholes_analyzer()", source)
        self.assertNotIn("jugaad", source.lower())
        self.assertNotIn("def bs_price(", source)
        self.assertNotIn("def bs_greeks(", source)

    def test_standalone_black_scholes_sidebar_page_is_removed(self):
        black_scholes_page = Path(__file__).resolve().parents[2] / "frontend" / "pages" / "6_Black_Scholes.py"
        self.assertFalse(black_scholes_page.exists())

    def test_backend_requirements_no_longer_include_jugaad(self):
        requirements = (Path(__file__).resolve().parents[2] / "backend" / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("jugaad", requirements.lower())


class LauncherScriptTests(SimpleTestCase):
    def test_macos_launchers_exist_and_reference_unix_start_scripts(self):
        repo_root = Path(__file__).resolve().parents[2]
        run_all_sh = (repo_root / "run_all.sh").read_text(encoding="utf-8")
        run_all_command = (repo_root / "run_all.command").read_text(encoding="utf-8")

        self.assertIn("scripts/start_django.sh", run_all_sh)
        self.assertIn("scripts/start_worker.sh", run_all_sh)
        self.assertIn("scripts/start_beat.sh", run_all_sh)
        self.assertIn("scripts/start_streamlit.sh", run_all_sh)
        self.assertIn("exec bash \"$REPO_DIR/run_all.sh\"", run_all_command)


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

    @patch("api.bl_numpy.fetch_benchmark_prices")
    @patch("api.bl_numpy.fetch_prices")
    def test_run_bl_analysis_returns_explicit_expected_and_realized_fields(
        self,
        mock_fetch_prices,
        mock_fetch_benchmark_prices,
    ):
        dates = pd.date_range("2025-01-01", periods=80, freq="B")
        price_frame = pd.DataFrame(
            {
                "AAPL": 100 * (1.0012 ** np.arange(len(dates))),
                "MSFT": 95 * (1.0009 ** np.arange(len(dates))),
            },
            index=dates,
        )
        benchmark_prices = pd.Series(
            400 * (1.0007 ** np.arange(len(dates))),
            index=dates,
            name="^GSPC",
        )
        mock_fetch_prices.return_value = price_frame
        mock_fetch_benchmark_prices.return_value = benchmark_prices

        result = run_bl_analysis(
            ["AAPL", "MSFT"],
            {"AAPL": 0.6, "MSFT": 0.4},
            [{"type": "absolute", "assets": ["AAPL"], "return_pct": 12.0, "confidence_pct": 70}],
            tau=0.05,
            risk_free_rate=0.02,
        )

        self.assertIsNone(result["error"])
        self.assertIn("benchmark_window", result)
        self.assertIn("realized_window", result)
        first_row = result["return_table"][0]
        self.assertIn("equilibrium_return_pct", first_row)
        self.assertIn("posterior_return_pct", first_row)
        self.assertIn("market_weight", first_row)
        self.assertIn("bl_weight", first_row)
        self.assertIn("weight_tilt", first_row)
        self.assertIn("excess_vs_benchmark_pct", first_row)
        self.assertAlmostEqual(
            first_row["weight_tilt"],
            round(first_row["bl_weight"] - first_row["market_weight"], 4),
        )
        self.assertEqual(result["benchmark_window"]["start"], dates.min().strftime("%Y-%m-%d"))
        self.assertEqual(result["benchmark_window"]["end"], dates.max().strftime("%Y-%m-%d"))
        self.assertEqual(result["realized_window"]["trading_days"], 30)

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

    @patch("api.ai_services.get_price_history")
    def test_optimize_changes_meaningfully_across_risk_levels(self, mock_get_price_history):
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        steps = np.arange(len(dates), dtype=float)
        histories = {
            "AAPL": 100 * np.cumprod(1 + 0.00055 + 0.00025 * np.sin(steps / 17)),
            "MSFT": 95 * np.cumprod(1 + 0.00045 + 0.00020 * np.cos(steps / 19)),
            "GOOG": 90 * np.cumprod(1 + 0.00100 + 0.00120 * np.sin(steps / 9)),
            "TSLA": 80 * np.cumprod(1 + 0.00190 + 0.00450 * np.sin(steps / 5)),
        }

        def fake_history(symbol, period="2y"):
            close = histories.get(symbol)
            if close is None:
                return pd.DataFrame()
            return pd.DataFrame({"Close": close}, index=dates)

        mock_get_price_history.side_effect = fake_history

        low = pd.Series(PortfolioOptimizer.optimize(["AAPL", "MSFT", "GOOG", "TSLA"], 0.0)["allocation"])
        mid = pd.Series(PortfolioOptimizer.optimize(["AAPL", "MSFT", "GOOG", "TSLA"], 0.5)["allocation"])
        high = pd.Series(PortfolioOptimizer.optimize(["AAPL", "MSFT", "GOOG", "TSLA"], 1.0)["allocation"])

        self.assertAlmostEqual(float(low.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(mid.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(high.sum()), 1.0, places=3)
        self.assertGreater(high["TSLA"], low["TSLA"])
        self.assertGreater(low["AAPL"], high["AAPL"])
        self.assertGreaterEqual(float((low - high).abs().max()), 0.05)
        self.assertGreaterEqual(float((low - high).abs().sum()), 0.20)

class BlackLittermanAllocatorTests(SimpleTestCase):
    @patch("api.bl_optimizer.build_views")
    @patch("api.bl_optimizer.get_market_caps")
    @patch("api.bl_optimizer.get_price_history")
    def test_run_black_litterman_changes_meaningfully_across_risk_levels(
        self,
        mock_get_price_history,
        mock_get_market_caps,
        mock_build_views,
    ):
        from .bl_optimizer import run_black_litterman

        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        steps = np.arange(len(dates), dtype=float)
        histories = {
            "AAPL": 100 * np.cumprod(1 + 0.00055 + 0.00025 * np.sin(steps / 17)),
            "MSFT": 95 * np.cumprod(1 + 0.00045 + 0.00020 * np.cos(steps / 19)),
            "GOOG": 90 * np.cumprod(1 + 0.00110 + 0.00120 * np.sin(steps / 8)),
            "TSLA": 80 * np.cumprod(1 + 0.00220 + 0.00500 * np.sin(steps / 4.5)),
        }

        def fake_history(symbol, period="2y"):
            close = histories.get(symbol)
            if close is None:
                return pd.DataFrame()
            return pd.DataFrame({"Close": close}, index=dates)

        mock_get_price_history.side_effect = fake_history
        mock_get_market_caps.return_value = {
            "AAPL": 2.8e12,
            "MSFT": 2.4e12,
            "GOOG": 1.8e12,
            "TSLA": 0.8e12,
        }
        mock_build_views.return_value = (
            {"AAPL": 0.07, "MSFT": 0.05, "GOOG": 0.13, "TSLA": 0.24},
            {"AAPL": 0.55, "MSFT": 0.45, "GOOG": 0.60, "TSLA": 0.70},
            [{"symbol": s} for s in ["AAPL", "MSFT", "GOOG", "TSLA"]],
        )

        low = run_black_litterman(["AAPL", "MSFT", "GOOG", "TSLA"], 0.0)
        mid = run_black_litterman(["AAPL", "MSFT", "GOOG", "TSLA"], 0.5)
        high = run_black_litterman(["AAPL", "MSFT", "GOOG", "TSLA"], 1.0)

        low_weights = pd.Series(low["allocation"])
        mid_weights = pd.Series(mid["allocation"])
        high_weights = pd.Series(high["allocation"])

        self.assertIsNone(low["error"])
        self.assertIsNone(mid["error"])
        self.assertIsNone(high["error"])
        self.assertAlmostEqual(float(low_weights.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(mid_weights.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(high_weights.sum()), 1.0, places=3)
        self.assertGreater(high_weights["TSLA"], low_weights["TSLA"])
        self.assertGreater(low_weights["AAPL"], high_weights["AAPL"])
        self.assertGreaterEqual(float((low_weights - high_weights).abs().max()), 0.05)
        self.assertGreaterEqual(float((low_weights - high_weights).abs().sum()), 0.20)

class BLOptimizationViewTests(SimpleTestCase):
    @patch("api.views._timed_call", return_value="")
    @patch("api.bl_optimizer.run_black_litterman")
    def test_bl_endpoint_passes_risk_tolerance(self, mock_run_black_litterman, _mock_timed_call):
        from .views import BLOptimizationView

        mock_run_black_litterman.return_value = {
            "allocation": {"AAPL": 0.5, "MSFT": 0.5},
            "view_details": [],
            "bl_returns": {"AAPL": 8.0, "MSFT": 6.0},
            "requested_tickers": ["AAPL", "MSFT"],
            "valid_tickers": ["AAPL", "MSFT"],
            "dropped_tickers": [],
            "view_tickers": ["AAPL", "MSFT"],
            "missing_view_tickers": [],
            "source": "bl",
            "blend_meta": {"from": "Min Vol", "to": "Max Sharpe", "alpha": 0.5},
            "fallback_reason": None,
            "error": None,
        }

        factory = APIRequestFactory()
        request = factory.post(
            "/api/bl-optimize/",
            {"tickers": ["AAPL", "MSFT"], "risk_tolerance": 0.75},
            format="json",
        )
        response = BLOptimizationView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        mock_run_black_litterman.assert_called_once_with(["AAPL", "MSFT"], 0.75)

