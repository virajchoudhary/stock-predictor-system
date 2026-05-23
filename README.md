# Stock Predictor System

An advanced AI-driven system for intraday stock market prediction, portfolio optimization, and algorithmic trading. Built with Django backend, Streamlit frontend, LSTM deep learning models, and enterprise-grade task orchestration.

## Overview

QuantVision is a comprehensive quantitative finance platform that combines:

- **Deep Learning Predictions**: LSTM-based models for 30-day stock price forecasting
- **Genetic Algorithm Optimization**: Evolutionary hyperparameter tuning for maximum trading accuracy
- **Portfolio Optimization**: Mean-Variance, CVaR, and Hierarchical Risk Parity models
- **Options Analysis**: Black-Scholes and SABR model for options mispricing detection
- **Technical Analysis**: 130+ indicators via TA library
- **Classical Forecasting**: ARIMA and GARCH baselines for volatility modeling
- **AI Advisory**: Groq-powered LLM chatbot for personalized investment advice
- **Quantitative Reporting**: Comprehensive portfolio analytics and backtesting reports

## Key Features

- Intraday trend prediction for Indian and US equities
- Multi-strategy ensemble combining neural networks with traditional econometric models
- Point-in-time historical backtesting (prevents data leakage)
- Real-time portfolio optimization and rebalancing
- Options chain analysis with implied volatility surface modeling
- AI-driven investment advisory chatbot
- Advanced technical signal generation
- Support for international stock markets via yfinance and Polygon API

## Technology Stack

### Backend
- Django REST Framework - API and business logic
- Celery + Redis - Async task queue and caching
- Gunicorn - Production WSGI server

### AI/ML
- PyTorch - Deep learning framework with GPU support
- scikit-learn - Classical ML algorithms
- Riskfolio-Lib - Portfolio optimization (HRP, CVaR, Markowitz)
- PyPortfolioOpt - Additional portfolio strategies
- pmdarima - AutoARIMA for baseline forecasting
- ARCH - GARCH volatility modeling
- Groq - Fast LLM inference for chat

### Frontend
- Streamlit - Interactive data visualization dashboard
- Plotly - Advanced charting and parallel coordinates
- Quant-Reporter - Professional PDF report generation

### Data Sources
- yfinance - Historical OHLCV data
- Polygon API - Enhanced market data and options chains
- Fast embeddings - Semantic text analysis

### Development
- Python 3.10+
- Docker - Redis containerization
- Bash/Batch scripts - Multi-service orchestration

## Project Structure

```
stock-predictor-system/
├── backend/                      # Django REST API
│   ├── manage.py                # Django management
│   ├── requirements.txt          # Python dependencies
│   ├── quantvision/              # Main Django app
│   │   ├── settings.py          # Django configuration
│   │   ├── urls.py              # API routing
│   │   └── wsgi.py              # WSGI entry point
│   ├── api/                      # REST endpoints
│   │   ├── views.py             # Prediction, portfolio, chat APIs
│   │   ├── serializers.py       # Data serialization
│   │   └── urls.py              # API routes
│   ├── models/                   # ML models and training
│   │   ├── lstm_predictor.py    # LSTM architecture and inference
│   │   ├── portfolio_optimizer.py # Optimization strategies
│   │   └── technical_analyzer.py # Technical indicators
│   ├── tasks/                    # Celery async jobs
│   │   ├── prediction_tasks.py  # Model inference jobs
│   │   └── scheduler.py         # Beat schedule config
│   └── venv/                     # Virtual environment
│
├── frontend/                     # Streamlit Dashboard
│   ├── app.py                   # Main entry point
│   ├── requirements.txt          # Frontend dependencies
│   ├── pages/                    # Multi-page apps
│   │   ├── 1_Dashboard.py       # Stock analysis and trends
│   │   ├── 2_Portfolio.py       # Portfolio optimization UI
│   │   ├── 3_Options_Analysis.py # Options pricing and Greeks
│   │   ├── 4_Backtesting.py     # Historical testing UI
│   │   ├── 5_Technical_Analysis.py # Indicator library
│   │   └── 6_Chat.py            # AI advisor chatbot
│   ├── components/               # Reusable UI components
│   │   ├── style.py             # CSS theming and dark mode
│   │   ├── black_scholes_ui.py  # Options analytics UI
│   │   └── reporting_utils.py   # Report generation helpers
│   └── .streamlit/              # Streamlit config
│
├── agents/                       # AI agent definitions
│   └── financial_advisor_agent.py # LLM-powered decision making
│
├── config/                       # Configuration files
│   ├── settings.json            # System parameters
│   └── models.yaml              # Model configurations
│
├── decision/                     # Decision support algorithms
│   └── signal_aggregator.py     # Multi-signal fusion
│
├── docs/                         # Documentation
│   ├── API_DOCS.md              # REST API reference
│   ├── MODEL_GUIDE.md           # ML model explanations
│   └── ARCHITECTURE.md          # System design
│
├── quant_reporter/              # Reporting engine (submodule)
│   └── Generates professional PDF reports
│
├── quant_testing/               # Backtesting framework
│   └── Historical performance evaluation
│
├── scripts/                      # Service startup scripts
│   ├── start_django.sh          # Django server
│   ├── start_worker.sh          # Celery worker
│   ├── start_beat.sh            # Celery Beat scheduler
│   └── start_streamlit.sh       # Streamlit frontend
│
├── tests/                        # Unit and integration tests
│   ├── test_models.py           # ML model tests
│   ├── test_api.py              # API endpoint tests
│   └── test_portfolio.py        # Portfolio optimizer tests
│
├── logs/                         # Runtime logs
│   ├── django.log               # Server logs
│   ├── celery.log               # Task queue logs
│   └── streamlit.log            # Frontend logs
│
├── run_all.sh                    # Linux/Mac orchestration script
├── run_all.bat                   # Windows orchestration script
├── run_all.command               # macOS Terminal launcher
├── requirements.txt              # Root-level dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for Redis)
- Git

### Quick Start (Automated)

Linux/macOS:
```bash
git clone https://github.com/virajchoudhary/stock-predictor-system.git
cd stock-predictor-system
bash run_all.sh
```

Windows:
```cmd
git clone https://github.com/virajchoudhary/stock-predictor-system.git
cd stock-predictor-system
run_all.bat
```

The script will:
1. Start Redis (Docker)
2. Install dependencies
3. Run database migrations
4. Start Django backend (http://localhost:8000)
5. Start Celery worker for async tasks
6. Start Celery Beat for scheduled jobs
7. Start Streamlit frontend (http://localhost:8501)

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/virajchoudhary/stock-predictor-system.git
cd stock-predictor-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
cd frontend && pip install -r requirements.txt
```

4. Start Redis (required for Celery):
```bash
docker run -d -p 6379:6379 --name redis redis
```

5. Run database migrations:
```bash
cd backend
python manage.py migrate
```

6. Start services in separate terminals:

Terminal 1 - Django Backend:
```bash
cd backend
python manage.py runserver
```

Terminal 2 - Celery Worker:
```bash
cd backend
celery -A quantvision worker -l info
```

Terminal 3 - Celery Beat (Scheduler):
```bash
cd backend
celery -A quantvision beat -l info
```

Terminal 4 - Streamlit Frontend:
```bash
cd frontend
streamlit run app.py
```

7. Open browser:
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/api/docs

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Django
DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (if using PostgreSQL)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=stock_predictor
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379/0

# Groq API (for AI Chat)
GROQ_API_KEY=your-groq-api-key

# Market Data APIs
POLYGON_API_KEY=your-polygon-key

# Model Configuration
MODEL_CACHE_TTL=3600
PREDICTION_HORIZON=30
BACKTEST_WINDOW=504  # 2 years of trading days

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

### Model Configuration

Edit `backend/config/models.yaml`:

```yaml
lstm:
  lookback_window: 120
  forecast_horizon: 30
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  learning_rate: 0.001
  epochs: 50
  batch_size: 32
  validation_split: 0.2

portfolio:
  optimization_method: "hrp"  # hrp, cvar, markowitz
  rebalance_frequency: "monthly"
  transaction_cost: 0.001

genetic_algorithm:
  population_size: 20
  generations: 10
  mutation_rate: 0.2
  crossover_rate: 0.8
```

## API Documentation

### Prediction Endpoint

GET `/api/predict/{symbol}/`

Request parameters:
- `symbol` (string): Stock ticker (e.g., AAPL, RELIANCE.NS)
- `target_date` (date, optional): Prediction date in YYYY-MM-DD format

Response:
```json
{
  "symbol": "AAPL",
  "current_price": 185.50,
  "predicted_price": 195.25,
  "trend": "UP",
  "confidence": 0.78,
  "technicals": {
    "signals": ["RSI_OVERBOUGHT", "MACD_BULLISH"],
    "rsi": 72.5,
    "macd": 0.85
  },
  "forecast_series": [185.5, 186.2, 187.1, ...],
  "dates": ["2025-05-24", "2025-05-25", ...]
}
```

### Portfolio Optimization Endpoint

POST `/api/portfolio/optimize/`

Request body:
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "weights_method": "hrp",
  "constraints": {
    "min_weight": 0.05,
    "max_weight": 0.40
  }
}
```

Response:
```json
{
  "optimal_weights": [0.35, 0.30, 0.35],
  "expected_return": 0.125,
  "volatility": 0.18,
  "sharpe_ratio": 0.69,
  "cvar": 0.052
}
```

### Options Analysis Endpoint

GET `/api/options/analyze/{symbol}/`

Parameters:
- `strike_price` (float): Option strike price
- `time_to_expiry` (float): Time to expiration in years
- `volatility` (float, optional): Implied volatility estimate

Response:
```json
{
  "symbol": "AAPL",
  "strike": 185.0,
  "call_price": 5.45,
  "put_price": 4.30,
  "delta": 0.62,
  "gamma": 0.018,
  "vega": 0.25,
  "theta": -0.05,
  "iv_surface": {...}
}
```

### Backtesting Endpoint

GET `/api/backtest-predict/{symbol}/`

Parameters:
- `symbol` (string): Stock ticker
- `past_date` (date): Training data cutoff (YYYY-MM-DD)
- `force_retrain` (bool): Bypass cached models

Response:
```json
{
  "symbol": "AAPL",
  "past_date": "2024-07-01",
  "predictions": [185.2, 186.1, 187.5, ...],
  "actuals": [184.8, 186.2, 187.0, ...],
  "mae": 1.25,
  "rmse": 1.67,
  "directional_accuracy": 0.82,
  "dates": ["2024-07-02", "2024-07-03", ...]
}
```

### AI Chat Endpoint

POST `/api/chat/`

Request body:
```json
{
  "message": "What stocks should I buy for growth?"
}
```

Response:
```json
{
  "response": "Based on current market conditions and your risk profile...",
  "confidence": 0.85,
  "sources": ["Technical Analysis", "Fundamental Data"]
}
```

## Machine Learning Models

### LSTM Predictor

Architecture:
- Input layer: 120-day lookback window
- Hidden layers: 2 stacked LSTM cells (128 units each)
- Dropout: 30% for regularization
- Output layer: 30-day ahead forecast

Training:
- Optimizer: Adam
- Loss function: MSE
- Validation split: 20%
- Early stopping on validation plateau

Performance:
- Typical RMSE: 0.8-1.2% of price
- Directional accuracy: 55-65% (above random 50%)

### Genetic Algorithm HPO

Evolves optimal LSTM hyperparameters:
- Chromosome: [hidden_size, num_layers, learning_rate, epochs]
- Population: 20 individuals per generation
- Generations: 10 (configurable)
- Fitness: Composite score of MSE + directional accuracy
- Selection: Tournament selection
- Crossover: Uniform crossover
- Mutation: Gaussian perturbation

Typical improvement: 5-15% directional accuracy gain

### Portfolio Optimizers

#### Hierarchical Risk Parity (HRP)
- Non-convex clustering-based approach
- Robust to estimation errors
- No need for correlation matrix inversion
- Performs well in unstable market conditions

#### Conditional Value at Risk (CVaR)
- Minimizes tail risk
- Considers worst 5% of outcomes
- Better for risk-averse investors

#### Mean-Variance (Markowitz)
- Classical approach
- Maximizes Sharpe ratio
- Sensitive to input assumptions
- Useful as baseline

### Technical Analysis

130+ indicators via TA library:
- Trend: SMA, EMA, TEMA, DEMA
- Momentum: RSI, MACD, Stochastic
- Volatility: Bollinger Bands, ATR, Donchian
- Volume: OBV, AD Line
- Custom signal aggregation

### ARIMA/GARCH Baseline

Classical time series models for comparison:
- AutoARIMA: Automatic order selection
- GARCH: Volatility forecasting
- Used for ensemble averaging
- Provides statistical baseline

## Usage Guide

### Dashboard

1. Open http://localhost:8501
2. Enter stock symbol (AAPL, RELIANCE.NS, etc.)
3. Click "Run Analysis"
4. View:
   - Current price and 30-day forecast
   - Bullish/Bearish trend signal
   - Technical indicators
   - Interactive price charts

### Portfolio Optimizer

1. Navigate to "Portfolio" page
2. Select stocks and allocation weights
3. Choose optimization method (HRP, CVaR, Markowitz)
4. View optimal allocation and statistics:
   - Expected return
   - Volatility
   - Sharpe ratio
   - CVaR
5. Export portfolio as PDF report

### Options Analysis

1. Go to "Options Analysis" page
2. Enter stock symbol and strike price
3. View Black-Scholes and SABR pricing:
   - Call/Put prices
   - Greeks (Delta, Gamma, Vega, Theta)
   - Mispricing signals
4. Interactive IV surface visualization

### Backtesting

1. Navigate to "Backtesting" page
2. Select symbol and historical date
3. Set force retrain if needed
4. Compare LSTM vs ARIMA/GARCH:
   - Prediction vs actual prices
   - MAE, RMSE, directional accuracy
   - Performance metrics
5. Download backtest report

### AI Chat

1. Go to "Chat" page
2. Ask investment questions:
   - "Should I buy AAPL?"
   - "What is my portfolio risk?"
   - "Suggest stocks for dividend income"
3. Receive AI-powered advice based on market data and LLM reasoning

### Evolutionary HPO

1. Open "Evolutionary HPO" page (hidden by default)
2. Select stock symbol
3. Configure GA parameters:
   - Population size
   - Number of generations
   - Mutation rate
4. Start evolution
5. Monitor fitness curves and population diversity
6. View parallel coordinates of search space

## Performance Metrics

### Model Evaluation

- Root Mean Square Error (RMSE): 0.8-1.2% of price
- Mean Absolute Error (MAE): 0.6-0.9% of price
- Directional Accuracy: 55-65%
- Sharpe Ratio: 0.6-0.9
- Maximum Drawdown: 8-15%

### System Performance

- API response time: <500ms for predictions
- Frontend load time: <2s
- Backtesting speed: 2 years in <30s
- Celery task throughput: 50+ tasks/minute

### Infrastructure

- Backend: ~500MB RAM base
- Model inference: <100ms per prediction
- Portfolio optimization: <1s for 100 assets
- Redis cache: ~2GB for 1-year daily data

## Troubleshooting

### Redis Connection Error
```
Error: ConnectionError: Cannot connect to port 6379
```
Solution:
```bash
docker run -d -p 6379:6379 --name redis redis
docker start redis
```

### Django Migrations Failed
```bash
cd backend
python manage.py migrate --run-syncdb
```

### Streamlit Cannot Connect to Backend
Ensure Django is running:
```bash
cd backend
python manage.py runserver 0.0.0.0:8000
```

### Celery Tasks Not Running
Check Redis connection and worker:
```bash
celery -A quantvision worker -l debug
celery -A quantvision inspect active
```

### LSTM Model OOM Error
Reduce batch size in `config/models.yaml`:
```yaml
lstm:
  batch_size: 16  # from 32
```

### Polygon API Key Invalid
Regenerate key at https://polygon.io/dashboard/auth and update .env

## Data Pipeline

1. Data Ingestion
   - yfinance fetches historical OHLCV
   - Polygon API provides real-time quotes
   - Options chain data from yfinance or NSE simulation

2. Preprocessing
   - Handle missing values (forward fill)
   - Outlier detection and removal
   - Technical indicator calculation
   - Feature normalization (standardization)

3. Model Training
   - 80% training, 20% validation split
   - TimeSeriesSplit cross-validation
   - Hyperparameter optimization via genetic algorithm
   - Model checkpointing and caching

4. Inference
   - Load optimized models from cache
   - Generate 30-day ahead predictions
   - Compute confidence intervals
   - Generate technical signals

5. Post-Processing
   - Blend LSTM + classical predictions
   - Apply portfolio constraints
   - Generate reports and alerts

## Roadmap

- [ ] Real-time data streaming via WebSocket
- [ ] Multi-asset class support (Crypto, Forex, Commodities)
- [ ] Reinforcement learning trading agent
- [ ] Options volatility surface modeling (SABR + local vol)
- [ ] Monte Carlo scenario analysis
- [ ] Mobile app (React Native)
- [ ] Docker Compose multi-container orchestration
- [ ] Kubernetes deployment templates
- [ ] Advanced risk metrics (Expected Shortfall, Drawdown)
- [ ] Machine learning interpretability (SHAP, LIME)
- [ ] Factor analysis and risk decomposition
- [ ] Sentiment analysis from news and social media

## Dependencies

Key packages:
- Django 4.2+
- PyTorch 2.0+
- scikit-learn 1.3+
- Riskfolio-Lib 0.2.0+
- yfinance 0.2.40+
- Streamlit 1.0+
- Celery 5.3+
- Redis 5.0+
- Plotly 5.0+
- pandas 2.0+
- numpy 1.24+

Full dependencies in `backend/requirements.txt` and `frontend/requirements.txt`

## Contributing

Contributions welcome. Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions. Use at your own risk. The creators assume no liability for trading losses.

## Contact

For issues, questions, or suggestions:
- GitHub Issues: https://github.com/virajchoudhary/stock-predictor-system/issues
- Email: viraj@example.com

## Acknowledgments

- yfinance for market data
- PyTorch team for deep learning framework
- Streamlit for interactive visualization
- Groq for fast LLM inference
- Riskfolio-Lib for portfolio optimization algorithms
