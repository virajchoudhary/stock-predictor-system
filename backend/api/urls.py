from django.urls import path
from .views import (ChatView, TrendPredictionView, PortfolioOptimizationView,
                    HPOStatusView, BLOptimizationView, BlackLittermanAnalysisView,
                    SWOTAnalysisView, train_rl_agent, rl_progress,
                    rl_hpo_cache_status, rl_hpo_cache_clear, stream_evolution,
                    BacktestPredictView)

urlpatterns = [
    path('chat/', ChatView.as_view(), name='chat'),
    path('predict/<str:symbol>/', TrendPredictionView.as_view(), name='predict_trend'),
    path('optimize/', PortfolioOptimizationView.as_view(), name='optimize_portfolio'),
    path('hpo-status/', HPOStatusView.as_view(), name='hpo-status'),
    path('bl-optimize/', BLOptimizationView.as_view(), name='bl_optimize'),
    path('black-litterman/', BlackLittermanAnalysisView.as_view(), name='black_litterman'),
    path('swot/', SWOTAnalysisView.as_view(), name='swot'),
    # RL Ensemble
    path('rl/train/', train_rl_agent, name='rl_train'),
    path('rl/progress/<str:session_id>/', rl_progress, name='rl_progress'),
    path('rl/hpo/status/', rl_hpo_cache_status, name='rl_hpo_status'),
    path('rl/hpo/clear/', rl_hpo_cache_clear, name='rl_hpo_clear'),
    path('evolution/', stream_evolution, name='evolution'),
    # Model Validation (backtest from past date to today)
    path('backtest-predict/<str:symbol>/', BacktestPredictView.as_view(), name='backtest_predict'),
]
