from django.urls import path
from .views import ChatView, TrendPredictionView, PortfolioOptimizationView

urlpatterns = [
    path('chat/', ChatView.as_view(), name='chat'),
    path('predict/<str:symbol>/', TrendPredictionView.as_view(), name='predict_trend'),
    path('optimize/', PortfolioOptimizationView.as_view(), name='optimize_portfolio'),
]
