# Git Commands to Push Your Code

Run these commands in your terminal (make sure you are in the `C:\QuantVis` directory):

```bash
# 1. Create a new branch and switch to it
git checkout -b feature/hpo-integration

# 2. Add all your changes
git add .

# 3. Commit the changes
git commit -m "feat: Integrate Evolutionary Hyperparameter Optimization (HPO)"

# 4. Push the branch to GitHub
git push -u origin feature/hpo-integration
```

***

# Pull Request (PR) Details to Copy & Paste

When you open the Pull Request on GitHub, use the following text. 
**Note:** Make sure to replace `[ISSUE_NUMBER]` with the actual number of the GitHub issue you opened previously (for example, `Closes #2`).

### **PR Title**
`feat: Add Evolutionary Hyperparameter Optimization (HPO) for LSTM`

### **PR Description**

```markdown
Closes #[ISSUE_NUMBER]

## Overview
This PR fully integrates an Evolutionary Genetic Algorithm (GA) to automatically optimize the hyperparameters of the PyTorch LSTM model used in our Market Trend Predictor. It directly connects the optimization process to real financial metrics (Directional Trading Accuracy) rather than just minimizing machine learning loss.

## Key Changes
- **Evolutionary Backend (`api/evolution.py`)**: Built a GA from scratch with a composite fitness function that penalizes high MSE and rewards Directional Accuracy (>50% edge).
- **Multi-Feature Data Pipeline**: Upgraded the LSTM's training data to include `Close`, `RSI`, `MACD`, `Bollinger Band Width`, `ATR`, and `SMA Distance` for a richer market context.
- **Model Persistence (`api/models.py`)**: Added an `OptimizedHyperparams` Django model that saves the best-evolved "chromosome" directly to the SQLite database with a 3-day cache window.
- **Dynamic Predictions (`api/ai_services.py` & `api/views.py`)**: `TrendPredictor.predict()` now seamlessly detects if an optimized model exists for a requested symbol and trains the LSTM with those specific hyperparameters instead of falling back to default settings.
- **Frontend Visualization (`pages/5_Evolutionary_Optimization.py`)**: Upgraded the Streamlit Dashboard. It now streams generations live, visually maps the search space using a parallel coordinates plot, highlights the 50% "profitable edge" barrier on the charts, and auto-checks the cache status.
- **Clean UI**: Removed non-professional UI elements and emojis across the entire platform.

## How to Test
1. Start the Django backend and Streamlit frontend.
2. Go to the **Evolutionary Optimization** page.
3. Check the status indicator (it will show that the default AI is being used for a given stock).
4. Run the Evolution for that stock. Watch the real-time Streamlit charts map validation loss against directional accuracy.
5. Once complete, navigate to the main **Dashboard** and request a prediction for that same stock.
6. Verify the badge explicitly states it is now "Powered by Evolved AI" and displays the evolved accuracy score.
```
