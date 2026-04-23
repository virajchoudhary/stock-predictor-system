TITLE:
Fix Deep Report Generation (quant-reporter Crash with yfinance >= 0.2.40)

DESCRIPTION:

### Problem
The "Deep Report" generation tab was throwing an ambiguous "Truth value of a Series" error when `quant-reporter` ran its background risk-free rate calculations and rolling return calculations.
This crash was introduced precisely because we recently bumped `yfinance` to `>=0.2.40` in `requirements.txt`. In these newer yfinance versions, queries frequently return a pandas `Series` instead of a plain `float`. The upstream `quant-reporter` package was not fully compatible with this `Series` return type.

### Proposed Change
Instead of downgrading `yfinance` (which would break our news parser) or forcing everyone to manually edit files in their `venv`, I have injected a lightweight monkey patch directly into `frontend/pages/3_Portfolio.py`. 
Whenever `quant-reporter` is loaded in the frontend, its sub-module `opt_core` functions (`get_risk_free_rate` and `calculate_rolling_returns`) are securely patched on-the-fly to handle pandas Series unpacking via `np.ravel()`. 

### Advantages
- The Deep Report now generates completely successfully without failing on multi-index or Series data.
- Everyone who pulls this branch gets the fix immediately; no manual `site-packages` tweaks are required for any team member!
- `yfinance` remains safely above `>=0.2.40` for our other micro-services.
