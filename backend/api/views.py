from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .ai_services import GroqService, TrendPredictor, PortfolioOptimizer, RLPortfolioAgent
from django.http import StreamingHttpResponse

_rl_agent = RLPortfolioAgent()


def _timed_call(func, timeout_seconds, fallback):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        return fallback
    except Exception:
        return fallback
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class ChatView(APIView):
    def post(self, request):
        message = request.data.get('message')
        if not message:
            return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        response = GroqService.chat(message)
        return Response({'response': response})

class TrendPredictionView(APIView):
    def get(self, request, symbol):
        target_date = request.query_params.get('target_date')
        from .models import OptimizedHyperparams
        cached = OptimizedHyperparams.get_valid_cache(symbol)

        if cached:
            prediction = TrendPredictor.predict(
                symbol,
                hidden_size        = cached.hidden_size,
                num_layers         = cached.num_layers,
                learning_rate      = cached.learning_rate,
                epochs             = cached.epochs,
                dropout            = cached.dropout,
                seq_len            = cached.seq_len,
                hyperparams_source = 'evolved',
                target_date        = target_date
            )
            prediction['evolved_accuracy'] = cached.directional_accuracy
        else:
            prediction = TrendPredictor.predict(symbol, target_date=target_date)

        from .models import SearchedTicker
        SearchedTicker.log(symbol)

        return Response(prediction)

class PortfolioOptimizationView(APIView):
    def post(self, request):
        tickers = request.data.get('tickers', [])
        risk_tolerance = request.data.get('risk_tolerance', 0.5)
        target_date = request.data.get('target_date')
        
        if not tickers:
            return Response({'error': 'Tickers are required'}, status=status.HTTP_400_BAD_REQUEST)
            
        # Auto-queue GA for any ticker without an evolved model
        try:
            from .tasks import queue_ga_for_ticker
            from .models import OptimizedHyperparams
            for ticker in tickers:
                if not OptimizedHyperparams.get_valid_cache(ticker):
                    queue_ga_for_ticker(ticker)
        except Exception:
            pass  # never block the optimization request

        optimization_result = PortfolioOptimizer.optimize(tickers, risk_tolerance, target_date=target_date)
        allocation = optimization_result.get("allocation", {})

        reasoning = _timed_call(
            lambda: GroqService.analyze_allocation(tickers, allocation, risk_tolerance),
            timeout_seconds=2.5,
            fallback="",
        )

        return Response({
            **optimization_result,
            'reasoning': reasoning
        })

class HPOStatusView(APIView):
    def get(self, request):
        from .models import OptimizedHyperparams
        symbol = request.query_params.get('symbol', 'AAPL').upper()
        cached = OptimizedHyperparams.get_valid_cache(symbol)
        if cached:
            return Response({
                'optimized':            True,
                'symbol':               symbol,
                'directional_accuracy': cached.directional_accuracy,
                'val_mse':              cached.val_mse,
                'hidden_size':          cached.hidden_size,
                'num_layers':           cached.num_layers,
                'learning_rate':        cached.learning_rate,
                'epochs':               cached.epochs,
                'cached_at':            cached.created_at.isoformat()
            })
        return Response({'optimized': False, 'symbol': symbol})

class BLOptimizationView(APIView):
    def post(self, request):
        tickers = request.data.get('tickers', [])
        target_date = request.data.get('target_date')
        try:
            risk_tolerance = float(request.data.get('risk_tolerance', 0.5))
        except (TypeError, ValueError):
            risk_tolerance = 0.5
        risk_tolerance = max(0.0, min(1.0, risk_tolerance))

        if not tickers or len(tickers) < 2:
            return Response(
                {'error': 'At least 2 tickers required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            from .bl_optimizer import run_black_litterman
        except ImportError:
            return Response(
                {'error': 'PyPortfolioOpt not installed. Run: pip install PyPortfolioOpt'},
                status=status.HTTP_400_BAD_REQUEST
            )

        result = run_black_litterman(tickers, risk_tolerance, target_date=target_date)

        if result.get("error"):
            return Response({'error': result["error"]}, status=status.HTTP_400_BAD_REQUEST)

        # Groq explanation
        prompt = f"""
        You are a portfolio manager. Explain the following Black-Litterman
        allocation in plain English in 3-4 sentences.

        Tickers: {tickers}
        Risk tolerance: {risk_tolerance:.2f} (0=conservative, 1=aggressive)

        LSTM-based views (what the AI predicts for each stock):
        {result['view_details']}

        Black-Litterman expected returns after blending with market equilibrium:
        {result['bl_returns']}

        Final allocation:
        {result['allocation']}

        Focus on: why the top weighted stocks got higher allocation,
        how the AI views influenced the result, and what the overall
        portfolio stance is (bullish/defensive/balanced).
        """

        reasoning = _timed_call(
            lambda: GroqService.chat(prompt),
            timeout_seconds=3.0,
            fallback="",
        )

        return Response({
            **result,
            'reasoning': reasoning
        })


class BlackLittermanAnalysisView(APIView):
    """
    Full Black-Litterman analysis with user-specified views.
    Implements BL math from scratch with NumPy.
    """
    def post(self, request):
        tickers = request.data.get('tickers', [])
        market_weights = request.data.get('market_weights', {})
        views = request.data.get('views', [])
        tau = request.data.get('tau', 0.05)
        risk_free_rate = request.data.get('risk_free_rate', 0.02)

        if not tickers or len(tickers) < 2:
            return Response(
                {'error': 'At least 2 tickers required.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not views:
            return Response(
                {'error': 'At least 1 view is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        from .bl_numpy import run_bl_analysis
        result = run_bl_analysis(tickers, market_weights, views, tau, risk_free_rate)

        if result.get("error"):
            return Response({'error': result["error"]}, status=status.HTTP_400_BAD_REQUEST)

        return Response(result)


class SWOTAnalysisView(APIView):
    def post(self, request):
        query = request.data.get('query', '')
        target_date = request.data.get('target_date')
        if not query:
            return Response({'error': 'Query is required'}, status=status.HTTP_400_BAD_REQUEST)

        from .swot_service import run_swot_analysis
        result = run_swot_analysis(query, target_date=target_date)

        if isinstance(result, dict) and result.get("error"):
            return Response({'error': result["error"]}, status=status.HTTP_400_BAD_REQUEST)

        return Response(result)


# ─────────────────────────────────────────────────────────────────────────────
# RL Ensemble — Train (async) + Progress poll + HPO cache
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_bool(value, default=True):
    if value is None: return default
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return bool(value)
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:  return True
        if value.strip().lower() in {"0", "false", "no", "off"}: return False
    return default


@api_view(['POST'])
def train_rl_agent(request):
    try:
        tickers    = request.data.get('tickers', [])
        timesteps  = int(request.data.get('timesteps', 10000))
        session_id = request.data.get('session_id', 'default')
        use_hpo    = _coerce_bool(request.data.get('use_hpo', True))
        target_date = request.data.get('target_date')
        if not tickers or len(tickers) < 2:
            return Response({'error': 'Provide at least 2 tickers.'}, status=400)

        def run_training():
            try:
                _rl_agent.train(tickers, timesteps=timesteps, session_id=session_id, use_hpo=use_hpo, target_date=target_date)
            except Exception as exc:
                import json as _json, os as _os
                pf = _os.path.join(_rl_agent.PROGRESS_DIR, f"{session_id}.json")
                with open(pf, 'w') as fh:
                    _json.dump({'progress': 0, 'status': 'error', 'error': str(exc)}, fh)

        threading.Thread(target=run_training, daemon=True).start()
        return Response({'status': 'started', 'session_id': session_id,
                         'hpo_enabled': use_hpo and timesteps >= _rl_agent.HPO_MIN_TIMESTEPS})
    except Exception as exc:
        return Response({'error': str(exc)}, status=500)


@api_view(['GET'])
def rl_progress(request, session_id):
    try:
        return Response(_rl_agent.get_progress(session_id))
    except Exception as exc:
        return Response({'error': str(exc)}, status=500)


@api_view(['GET'])
def rl_hpo_cache_status(request):
    try:
        tickers = [t.strip() for t in request.query_params.get('tickers', '').split(',') if t.strip()]
        if len(tickers) < 2:
            return Response({'error': 'Provide at least 2 tickers.'}, status=400)
        cached = _rl_agent._load_hpo_cache(tickers)
        if cached:
            return Response({'cached': True, 'ticker_hash': _rl_agent._ticker_hash_for_response(tickers),
                             'agents_evolved': list(cached.keys()), 'evolved_hyperparams': cached})
        return Response({'cached': False, 'tickers': tickers})
    except Exception as exc:
        return Response({'error': str(exc)}, status=500)


@api_view(['DELETE'])
def rl_hpo_cache_clear(request):
    try:
        tickers = request.data.get('tickers', [])
        if len(tickers) < 2:
            return Response({'error': 'Provide at least 2 tickers.'}, status=400)
        import os as _os
        path = _rl_agent._hpo_cache_path(tickers)
        if _os.path.exists(path):
            _os.remove(path)
            return Response({'status': 'cleared', 'tickers': tickers})
        return Response({'status': 'no_cache', 'tickers': tickers})
    except Exception as exc:
        return Response({'error': str(exc)}, status=500)

def stream_evolution(request):
    symbol = request.GET.get('symbol', 'AAPL')
    try:
        pop_size = int(request.GET.get('pop_size', 5))
    except ValueError:
        pop_size = 5
    try:
        generations = int(request.GET.get('generations', 5))
    except ValueError:
        generations = 5
    try:
        mutation_rate = float(request.GET.get('mutation_rate', 0.2))
    except ValueError:
        mutation_rate = 0.2
    target_date = request.GET.get('target_date')

    from .evolution import evolutionary_hpo_generator
    return StreamingHttpResponse(
        evolutionary_hpo_generator(
            symbol=symbol,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            target_date=target_date
        ),
        content_type='application/x-ndjson'
    )
