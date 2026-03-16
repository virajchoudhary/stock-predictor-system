from django.urls import path
from .views import ChatView, TrendPredictionView, PortfolioOptimizationView, train_rl_agent, rl_progress

urlpatterns = [
    path('rl/train/', train_rl_agent, name='rl_train'),
    path('rl/progress/<str:session_id>/', rl_progress, name='rl_progress'),
    path('chat/', ChatView.as_view(), name='chat'),
    path('predict/<str:symbol>/', TrendPredictionView.as_view(), name='predict_trend'),
    path('optimize/', PortfolioOptimizationView.as_view(), name='optimize_portfolio'),
]
