# Stock Predictor System

Minimal scaffold for a multi-workspace stock prediction system.

Structure created:

- `config/` - example environment variables
- `agents/scheduler/` - FastAPI scheduler + Redis producer
- `agents/predictor/` - Redis consumer placeholder (prediction)
- `agents/memory/` - MongoDB helper for memory
- `frontend/` - Streamlit demo app

Quick start (development):

1. Create a virtualenv and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Start Redis and MongoDB (e.g., via Docker), then run the scheduler API:

```bash
uvicorn agents.scheduler.main:app --reload --port 8000
```

3. Run the Streamlit frontend:

```bash
streamlit run frontend/app.py
```
