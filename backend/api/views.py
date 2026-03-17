from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse
import threading

from .ai_services import GroqService, TrendPredictor, PortfolioOptimizer

# ── Shared RL agent instance ──────────────────────────────────────────────────
from .ai_services import RLPortfolioAgent
_rl_agent = RLPortfolioAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatView(APIView):
    def post(self, request):
        message = request.data.get('message')
        if not message:
            return Response({'error': 'Message is required'},
                            status=status.HTTP_400_BAD_REQUEST)
        response = GroqService.chat(message)
        return Response({'response': response})


# ─────────────────────────────────────────────────────────────────────────────
# Trend Prediction  (uses evolved LSTM hyperparams if available)
# ─────────────────────────────────────────────────────────────────────────────

class TrendPredictionView(APIView):
    def get(self, request, symbol):
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
                hyperparams_source = 'evolved'
            )
            prediction['evolved_accuracy'] = cached.directional_accuracy
        else:
            prediction = TrendPredictor.predict(symbol)

        from .models import SearchedTicker
        SearchedTicker.log(symbol)

        return Response(prediction)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Optimisation (Riskfolio-Lib)
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizationView(APIView):
    def post(self, request):
        tickers        = request.data.get('tickers', [])
        risk_tolerance = request.data.get('risk_tolerance', 0.5)

        if not tickers:
            return Response({'error': 'Tickers are required'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Auto-queue GA for any ticker without an evolved LSTM model
        try:
            from .tasks import run_ga_for_ticker
            from .models import OptimizedHyperparams
            for ticker in tickers:
                if not OptimizedHyperparams.get_valid_cache(ticker):
                    run_ga_for_ticker.delay(ticker)
        except Exception:
            pass

        allocation = PortfolioOptimizer.optimize(tickers, risk_tolerance)
        reasoning  = GroqService.analyze_allocation(tickers, allocation, risk_tolerance)

        return Response({'allocation': allocation, 'reasoning': reasoning})


# ─────────────────────────────────────────────────────────────────────────────
# LSTM HPO Status
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Black-Litterman Optimisation
# ─────────────────────────────────────────────────────────────────────────────

class BLOptimizationView(APIView):
    def post(self, request):
        tickers = request.data.get('tickers', [])
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

        result = run_black_litterman(tickers)

        if result.get("error"):
            return Response({'error': result["error"]},
                            status=status.HTTP_400_BAD_REQUEST)

        prompt = f"""
        You are a portfolio manager. Explain this Black-Litterman allocation in
        3-4 sentences.

        Tickers: {tickers}
        LSTM-based views (AI predictions): {result['view_details']}
        BL expected returns after blending with market equilibrium: {result['bl_returns']}
        Final allocation: {result['allocation']}

        Focus on: why top-weighted stocks received higher allocation, how AI
        views influenced the result, and the overall portfolio stance.
        """
        reasoning = GroqService.chat(prompt)

        return Response({
            'allocation':   result['allocation'],
            'view_details': result['view_details'],
            'bl_returns':   result['bl_returns'],
            'reasoning':    reasoning
        })


# ─────────────────────────────────────────────────────────────────────────────
# RL Ensemble — Train (async) + Progress poll
# ─────────────────────────────────────────────────────────────────────────────

@api_view(['POST'])
def train_rl_agent(request):
    """
    Start RL ensemble training in a background thread.
    Body: { tickers, timesteps, session_id }

    If timesteps >= 10000, GA HPO runs first (per-agent, 4×3 evaluations),
    then the full A2C + SAC + TD3 training uses the evolved hyperparameters.
    Results are cached by ticker-set hash so HPO is not repeated unnecessarily.
    """
    try:
        tickers    = request.data.get('tickers', [])
        timesteps  = int(request.data.get('timesteps', 10000))
        session_id = request.data.get('session_id', 'default')

        if not tickers or len(tickers) < 2:
            return Response({'error': 'Provide at least 2 tickers.'}, status=400)

        def run_training():
            try:
                _rl_agent.train(tickers, timesteps=timesteps, session_id=session_id)
            except Exception as e:
                import os
                progress_file = os.path.join(_rl_agent.PROGRESS_DIR, f"{session_id}.json")
                import json
                with open(progress_file, 'w') as f:
                    json.dump({'progress': 0, 'status': 'error', 'error': str(e)}, f)

        threading.Thread(target=run_training, daemon=True).start()
        return Response({
            'status':     'started',
            'session_id': session_id,
            'hpo_enabled': timesteps >= _rl_agent.HPO_MIN_TIMESTEPS,
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def rl_progress(request, session_id):
    """Poll progress for a training session."""
    try:
        return Response(_rl_agent.get_progress(session_id))
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def rl_hpo_cache_status(request):
    """
    Check whether evolved RL hyperparams exist for a given ticker set.
    Query param: ?tickers=AAPL,MSFT,TCS.NS,RELIANCE.NS
    """
    try:
        tickers_str = request.query_params.get('tickers', '')
        tickers     = [t.strip() for t in tickers_str.split(',') if t.strip()]
        if len(tickers) < 2:
            return Response({'error': 'Provide at least 2 tickers.'}, status=400)

        cached = _rl_agent._load_hpo_cache(tickers)
        if cached:
            return Response({
                'cached':               True,
                'ticker_hash':          _rl_agent._ticker_hash_for_response(tickers),
                'agents_evolved':       list(cached.keys()),
                'evolved_hyperparams':  cached,
            })
        return Response({'cached': False, 'tickers': tickers})
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['DELETE'])
def rl_hpo_cache_clear(request):
    """
    Clear the HPO cache for a given ticker set (forces re-evolution on next train).
    Body: { tickers: [...] }
    """
    try:
        tickers = request.data.get('tickers', [])
        if len(tickers) < 2:
            return Response({'error': 'Provide at least 2 tickers.'}, status=400)
        path = _rl_agent._hpo_cache_path(tickers)
        import os
        if os.path.exists(path):
            os.remove(path)
            return Response({'status': 'cleared', 'tickers': tickers})
        return Response({'status': 'no_cache', 'tickers': tickers})
    except Exception as e:
        return Response({'error': str(e)}, status=500)
