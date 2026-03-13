# QuantVis

A full-stack financial analysis platform combining quantitative finance, deep learning, and LLM-powered reasoning. Built on Django (backend) and Streamlit (frontend).

---

## Features

- Live candlestick charts with MACD, RSI, and Bollinger Band overlays
- Portfolio optimization via Hierarchical Risk Parity, Min-CVaR, and Max-Sharpe using riskfolio-lib
- SABR volatility model calibration with interactive 3D volatility surfaces for options mispricing detection
- PyTorch LSTM with evolutionary hyperparameter optimization (Genetic Algorithm) for directional price prediction
- Groq LLM integration (Llama 3) for contextual market reasoning and portfolio analysis
- Exhaustive HTML backtest reports via quant_reporter (Monte Carlo, Walk-forward)

---

## Stack

| Layer    | Tech                                              |
|----------|---------------------------------------------------|
| Backend  | Django REST Framework                             |
| Frontend | Streamlit + Plotly                                |
| ML       | PyTorch, scikit-learn, ta                         |
| Quant    | riskfolio-lib, quant_reporter                     |
| Data     | Polygon.io (yfinance fallback)                    |
| AI       | Groq API (Llama 3.3 70B)                         |

---

## Setup

Requires Python 3.12+. A Groq API key is needed for AI features.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```env
GROQ_API_KEY=your_key_here
```

Run migrations:

```bash
python manage.py migrate
```

---

## Running

Two terminals are required.

**Terminal 1 — Backend:**
```bash
cd backend
source venv/bin/activate
python manage.py runserver 0.0.0.0:8000
```

**Terminal 2 — Frontend** (from project root):
```bash
./backend/venv/bin/streamlit run frontend/app.py --server.port 8501
```

Backend available at `http://localhost:8000`. Frontend at `http://localhost:8501`.

---

## Notes

- If `portfolio_report.html` throws a FileNotFoundError, make sure Streamlit is launched from the project root, not from inside `backend/`.
- Options data for Indian indices (`^NSEI`) is frequently blocked by yfinance. The app falls back to a synthetic SABR volatility smile in that case. Use `SPY` for live US options data.
- If a module is missing, confirm the venv is activated in that terminal before running.
