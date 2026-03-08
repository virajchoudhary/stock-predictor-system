from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
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
