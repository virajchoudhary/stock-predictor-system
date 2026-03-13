from django.urls import path
from .views import ChatView, TrendPredictionView, PortfolioOptimizationView, EvolutionHPOView, HPOStatusView

urlpatterns = [
    path('chat/', ChatView.as_view(), name='chat'),
    path('predict/<str:symbol>/', TrendPredictionView.as_view(), name='predict_trend'),
    path('optimize/', PortfolioOptimizationView.as_view(), name='optimize_portfolio'),
    path('evolution/', EvolutionHPOView.as_view(), name='evolution_hpo'),
    path('hpo-status/', HPOStatusView.as_view(), name='hpo-status'),
]
