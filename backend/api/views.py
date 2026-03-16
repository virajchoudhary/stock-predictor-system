from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
import threading
from rest_framework import status
from .ai_services import RLPortfolioAgent
_rl_agent = RLPortfolioAgent()
from .ai_services import GroqService, TrendPredictor, PortfolioOptimizer

class ChatView(APIView):
    def post(self, request):
        message = request.data.get('message')
        if not message:
            return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        response = GroqService.chat(message)
        return Response({'response': response})

class TrendPredictionView(APIView):
    def get(self, request, symbol):
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


@api_view(['POST'])
def train_rl_agent(request):
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
                import json, os
                progress_file = os.path.join(_rl_agent.PROGRESS_DIR, f"{session_id}.json")
                with open(progress_file, 'w') as f:
                    json.dump({'progress': 0, 'status': 'error', 'error': str(e)}, f)
        threading.Thread(target=run_training, daemon=True).start()
        return Response({'status': 'started', 'session_id': session_id})
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def rl_progress(request, session_id):
    try:
        return Response(_rl_agent.get_progress(session_id))
    except Exception as e:
        return Response({'error': str(e)}, status=500)
