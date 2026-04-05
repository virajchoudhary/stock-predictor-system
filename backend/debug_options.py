import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime

print("Starting Options Debug...")

def fetch_option_chain(symbol):
    print(f"Fetching chain for {symbol}")
    ticker = yf.Ticker(symbol)
    try:
        expirations = ticker.options
        if not expirations:
            print("No expirations found.")
            return None, None, None
        
        expiry = expirations[0] 
        print(f"Expiry: {expiry}")
        chain = ticker.option_chain(expiry)
        
        hist = ticker.history(period="1d")
        if hist.empty:
             print("No history found.")
             return chain, None, expiry
             
        current_price = hist['Close'].iloc[-1]
        print(f"Spot: {current_price}")
        
        return chain, current_price, expiry
    except Exception as e:
        print(f"Fetch Error: {e}")
        return None, None, None

def saber_vol(k, f, t, alpha, beta, rho, nu):
    # (Copied exactly from source)
    try:
        if k <= 0 or f <= 0 or t <= 0: return 0
        log_fk = np.log(f / k)
        fk_beta = (f * k) ** ((1 - beta) / 2)
        z = (nu / alpha) * fk_beta * log_fk
        if abs(z) < 1e-5:
            x_z = 1
        else:
            arg = 1 - 2 * rho * z + z * z
            if arg < 0: arg = 0
            x_z = np.log((np.sqrt(arg) + z - rho) / (1 - rho)) / z
        
        term1 = (1 - beta)**2 / 24 * log_fk**2
        term2 = (rho * beta * nu * alpha) / (4 * fk_beta)
        term3 = (2 - 3 * rho**2) * nu**2 / 24
        brackets = 1 + (term1 + term2 + term3) * t
        vol = (alpha / fk_beta) * (z / x_z if abs(x_z) > 1e-5 else 1) * brackets
        return vol
    except Exception:
        return 0

def calibrate_sabr(strikes, market_ivs, f, t):
    beta = 0.5
    
    def objective(params):
        alpha, rho, nu = params
        sabr_ivs = [saber_vol(k, f, t, alpha, beta, rho, nu) for k in strikes]
        # Check for NaNs in result
        sabr_ivs = np.array(sabr_ivs)
        if np.isnan(sabr_ivs).any():
            return 1e6 # Penalty
        error = np.sum((sabr_ivs - np.array(market_ivs)) ** 2)
        return error

    atm_vol = np.mean(market_ivs)
    initial_guess = [atm_vol, 0.0, 0.5]
    bounds = [(0.01, 2.0), (-0.99, 0.99), (0.01, 5.0)]
    
    print(f"Starting Minimize. Init: {initial_guess}")
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    print(f"Minimize Result: {result.message}")
    if not result.success:
        print(f"Minimize Failed: {result}")
    return result.x

# Run Logic
try:
    symbol = "^NSEI"
    chain, spot_price, expiry_date = fetch_option_chain(symbol)
    if chain is None:
        print("Failed to get chain")
        exit()

    calls = chain.calls
    upper_bound = spot_price * 1.10
    lower_bound = spot_price * 0.90
    liquid_calls = calls[(calls['strike'] > lower_bound) & (calls['strike'] < upper_bound)].copy()
    liquid_calls = liquid_calls[liquid_calls['impliedVolatility'] > 0.05]
    
    if liquid_calls.empty:
        print("No liquid calls")
        exit()
        
    expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    t = max((expiry_dt - datetime.now()).days / 365.0, 0.001)
    
    strikes = liquid_calls['strike'].values
    market_ivs = liquid_calls['impliedVolatility'].values
    
    # Valid Mask Logic
    valid_mask = np.isfinite(market_ivs) & (market_ivs > 0)
    print(f"Valid points: {np.sum(valid_mask)}")
    
    strikes = strikes[valid_mask]
    market_ivs = market_ivs[valid_mask]
    
    alpha, rho, nu = calibrate_sabr(strikes, market_ivs, spot_price, t)
    print(f"Calibrated: {alpha}, {rho}, {nu}")

except Exception as e:
    print("CRASHED!")
    import traceback
    traceback.print_exc()
