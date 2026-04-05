from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .ai_services import GroqService, TrendPredictor, PortfolioOptimizer
from django.http import StreamingHttpResponse
from .evolution import evolutionary_hpo_generator

class ChatView(APIView):
    def post(self, request):
        message = request.data.get('message')
        if not message:
            return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        response = GroqService.chat(message)
        return Response({'response': response})

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
                hyperparams_source = 'evolved'
            )
            prediction['evolved_accuracy'] = cached.directional_accuracy
        else:
            prediction = TrendPredictor.predict(symbol)

        return Response(prediction)

class PortfolioOptimizationView(APIView):
    def post(self, request):
        tickers = request.data.get('tickers', [])
        risk_tolerance = request.data.get('risk_tolerance', 0.5)
        
        if not tickers:
            return Response({'error': 'Tickers are required'}, status=status.HTTP_400_BAD_REQUEST)
            
        allocation = PortfolioOptimizer.optimize(tickers, risk_tolerance)
        
        # New: Get AI Commentary
        reasoning = GroqService.analyze_allocation(tickers, allocation, risk_tolerance)
        
        return Response({
            'allocation': allocation,
            'reasoning': reasoning
        })

class EvolutionHPOView(APIView):
    def get(self, request):
        symbol = request.query_params.get('symbol', 'AAPL')
        pop_size = int(request.query_params.get('pop_size', 5))
        generations = int(request.query_params.get('generations', 5))
        mutation_rate = float(request.query_params.get('mutation_rate', 0.2))
        
        generator = evolutionary_hpo_generator(
            symbol=symbol,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        return StreamingHttpResponse(generator, content_type='text/event-stream')

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


class BlackScholesView(APIView):
    """
    POST /api/black-scholes/
    Accepts option parameters and returns full Black–Scholes analysis.
    """
    def post(self, request):
        from .black_scholes import analyze_option

        required = ['S', 'K', 'T', 'r', 'sigma']
        missing = [f for f in required if f not in request.data]
        if missing:
            return Response(
                {'error': f'Missing required fields: {", ".join(missing)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = analyze_option(
                S=float(request.data['S']),
                K=float(request.data['K']),
                T=float(request.data['T']),
                r=float(request.data['r']),
                sigma=float(request.data['sigma']),
                q=float(request.data.get('q', 0.0)),
                market_price=float(request.data['market_price'])
                    if request.data.get('market_price') else None,
                option_type=request.data.get('option_type', 'call'),
                signal_threshold=float(request.data.get('signal_threshold', 5.0)),
            )
            return Response(result)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {'error': f'Computation failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SWOTView(APIView):
    """
    POST /api/swot/
    Full SWOT analysis pipeline: fetch real-time data + Groq LLM synthesis.
    """
    def post(self, request):
        from .swot_service import run_swot_analysis
        query = request.data.get('query', '').strip()
        if not query:
            return Response(
                {'error': 'Company name or ticker is required'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            result = run_swot_analysis(query)
            if 'error' in result and len(result) <= 2:
                return Response(result, status=status.HTTP_404_NOT_FOUND)
            return Response(result)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
