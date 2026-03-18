# QuantVis 📈

QuantVis is a comprehensive, advanced financial analysis and visualization platform. It combines a robust Django backend for data processing and AI integrations with an interactive Streamlit frontend for dynamic portfolio optimization, market analysis, options mispricing detection, and reporting.

---

## 🏗️ Project Structure

The project is divided into two primary components:

*   **`/backend`**: A Django application responsible for API endpoints, AI integrations (e.g., GroqLLM), portfolio optimization logic, and heavy data computations (e.g., `quant_reporter`).
*   **`/frontend`**: A Streamlit application that consumes the backend APIs to provide interactive dashboards, volatility surfaces, and detailed HTML reports.

Both components share the same Python virtual environment located in `/backend/venv`.

---

## ⚙️ Prerequisites

1.  **Python 3.12+**
2.  **API Keys**: You will need a Groq API Key for the AI contextual chat and reasoning features.

---

## 🚀 Setup & Installation

### 1. Initialize the Environment
Open a terminal and navigate to the `backend` directory to set up the virtual environment:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables (.env)
Create a `.env` file inside the `backend/` directory to store your secrets. It should look like this:

```env
GROQ_API_KEY="your_groq_api_key_here"
# Add other environment variables if necessary
```

### 3. Database Migrations (Backend)
Ensure the Django database is set up:
```bash
python manage.py migrate
```

---

## 🏃‍♂️ How to Run the Application

You will need **two separate terminal windows/tabs** to run the backend and frontend simultaneously.

### Terminal 1: Start the Django Backend Server

The backend must be running for the frontend to fetch optimization data and AI reasoning.

```bash
cd backend
source venv/bin/activate
python manage.py runserver 0.0.0.0:8000
```
*The backend API will be available at `http://localhost:8000`.*

### Terminal 2: Start the Streamlit Frontend

Open a new terminal window at the **root of the `QuantVis` project** (not inside the backend folder). We will use the backend's virtual environment to launch Streamlit.

```bash
# Ensure you are in the root directory: c:\stock price predictor
./backend/venv/bin/streamlit run frontend/app.py --server.port 8501
```

*The interactive dashboard will automatically open in your browser at `http://localhost:8501`.*

---

## 🌟 Key Features

1.  **Dashboard Hub**: Live market data, dynamically scaling Plotly candlestick charts, MACD/RSI technical indicators, and an AI-powered contextual chat assistant.
2.  **Portfolio Manager**: 
    *   **Allocator:** Mean-Variance optimization with AI-generated reasoning on the suggested allocations.
    *   **Deep Report:** Generates exhaustive HTML backtest reports combining Monte Carlo simulations, Walk-forward analysis, and risk metrics using `quant_reporter`.
3.  **Options Mispricing Detector**: Uses the SABR Volatility model to calibrate implied volatilities against the market, identifying statistically overpriced or undervalued options contracts. Includes interactive 3D Volatility Surfaces. 

---

## ⚠️ Troubleshooting

1.  **`FileNotFoundError: portfolio_report.html`**: Ensure you are running the Streamlit app from the root directory so the relative paths resolve correctly.
2.  **Options Analysis "Simulation Mode"**: `yfinance` frequently blocks live Option Chains for Indian indices (e.g. `^NSEI`). If the app cannot fetch live data via `nsepython` or `yfinance`, it will gracefully fall back to a "Simulation Mode" generating a synthetic volatility smile to demonstrate the SABR calibration logic safely. Try symbol `SPY` to see live US Market option data.
3.  **No Module Named 'X'**: Ensure your `venv` is activated in that specific terminal before running the server or streamlit.
