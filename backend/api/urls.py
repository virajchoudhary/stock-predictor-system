from django.urls import path
from .views import ChatView, TrendPredictionView, PortfolioOptimizationView, HPOStatusView, BLOptimizationView, BlackLittermanAnalysisView

urlpatterns = [
    path('chat/', ChatView.as_view(), name='chat'),
    path('predict/<str:symbol>/', TrendPredictionView.as_view(), name='predict_trend'),
    path('optimize/', PortfolioOptimizationView.as_view(), name='optimize_portfolio'),
    path('hpo-status/', HPOStatusView.as_view(), name='hpo-status'),
    path('bl-optimize/', BLOptimizationView.as_view(), name='bl_optimize'),
    path('black-litterman/', BlackLittermanAnalysisView.as_view(), name='black_litterman'),
]
