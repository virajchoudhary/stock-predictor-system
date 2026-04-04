from django.urls import path
from .views import (
    ChatView,
    TrendPredictionView,
    PortfolioOptimizationView,
    HPOStatusView,
    BLOptimizationView,
    train_rl_agent,
    rl_progress,
    rl_hpo_cache_status,
    rl_hpo_cache_clear,
)

urlpatterns = [
    # Chat
    path('chat/', ChatView.as_view(), name='chat'),

    # Trend prediction (uses evolved LSTM hyperparams if available)
    path('predict/<str:symbol>/', TrendPredictionView.as_view(), name='predict_trend'),

    # Riskfolio portfolio optimisation
    path('optimize/', PortfolioOptimizationView.as_view(), name='optimize_portfolio'),

    # LSTM HPO status
    path('hpo/status/', HPOStatusView.as_view(), name='hpo_status'),

    # Black-Litterman optimisation
    path('bl/optimize/', BLOptimizationView.as_view(), name='bl_optimize'),

    # RL ensemble — train + progress
    path('rl/train/', train_rl_agent, name='rl_train'),
    path('rl/progress/<str:session_id>/', rl_progress, name='rl_progress'),

    # RL HPO cache management
    path('rl/hpo/status/', rl_hpo_cache_status, name='rl_hpo_status'),
    path('rl/hpo/clear/', rl_hpo_cache_clear, name='rl_hpo_clear'),
]
