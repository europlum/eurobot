import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import dawsn
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config  # noqa: E402 # type: ignore

def check_structural_break_cusum(residuals, significance_level=0.05):
    n = len(residuals)
    sigma = np.std(residuals, ddof=1)
    if sigma == 0: 
        return True 
    cusum_path = np.cumsum(residuals) / (sigma * np.sqrt(n))
    test_stat = np.max(np.abs(cusum_path))
    if test_stat > 1.358:
        return True
    return False

def fit_ou_parameters(spread_series, dt=1.0):
    X = spread_series.to_numpy() if hasattr(spread_series, 'to_numpy') else spread_series
    y_arr = X[1:]
    x_arr = X[:-1]
    
    poly = np.polyfit(x_arr, y_arr, deg=1)
    a, b = poly[0], poly[1]
    if a <= 0 or a >= 1:
        return None
        
    residuals = y_arr - (a * x_arr + b)
    if check_structural_break_cusum(residuals):
        return None 
    res_var = np.var(residuals, ddof=2)
    
    mu = -np.log(a) / dt
    theta = b / (1 - a)
    sigma = np.sqrt(res_var * 2 * mu / max((1 - a**2), 1e-8))
    if mu <= 1e-8:
        return None
        
    half_life = np.log(2) / mu
    if half_life > config.MAX_HALF_LIFE_BARS:
        return None
    return {"mu": mu, "theta": theta, "sigma": sigma, "half_life": half_life}

def calculate_optimal_thresholds(ou_params):
    mu, theta, sigma = ou_params["mu"], ou_params["theta"], ou_params["sigma"]
    round_trip_cost = config.TRANSACTION_COST * 4 
    
    asymptotic_std = sigma / np.sqrt(2 * mu)
    
    if (asymptotic_std * 2) <= round_trip_cost:
        return None

    def expected_passage_time(entry_price):
        x0 = entry_price - theta
        k = mu / sigma**2
        sqrt_k = np.sqrt(k)
        
        def constant_integrand(z):
            return np.exp(np.clip(k * z**2, None, 700.0))
        
        C, _ = quad(constant_integrand, -10, 0)
        
        def outer_integrand(y):
            part1 = (1.0 / sqrt_k) * dawsn(y * sqrt_k)
            part2 = C * np.exp(np.clip(-k * y**2, -700.0, None))
            return part1 + part2
            
        time, _ = quad(outer_integrand, x0, 0)
        if time > 1e300 or np.isinf(time): 
            return 1e300
        return (2 / sigma**2) * time

    def objective_function(entry_price):
        expected_time = expected_passage_time(entry_price)
        if expected_time <= 0 or np.isinf(expected_time): 
            return 1e9
        
        expected_profit = (theta - entry_price) - round_trip_cost

        if expected_profit <= 1e-6: 
            return 1e9
            
        return -(expected_profit / expected_time)

    search_bound_high = theta - round_trip_cost - (asymptotic_std * 0.05)
    search_bound_low = theta - (asymptotic_std * 5.0)      

    if search_bound_low >= search_bound_high:
        return None

    result = minimize_scalar(
        objective_function, bounds=(search_bound_low, search_bound_high), method='bounded'
    )
    
    if not result.success:
        return None
        
    optimal_entry = result.x
    spread_width = abs(theta - optimal_entry)
    
    return {
        "long_entry": theta - spread_width,
        "long_exit": (theta - spread_width) + (spread_width * config.EXIT_TARGET_PCT),
        "short_entry": theta + spread_width,
        "short_exit": (theta + spread_width) - (spread_width * config.EXIT_TARGET_PCT)
    }

def process_window(job):
    pair_key, effective_date = job["pair_key"], job["effective_date"]
    pair_key
    ticker_a, ticker_b = job["ticker_a"], job["ticker_b"]
    ou_metrics = fit_ou_parameters(job["spread_data"])
    
    if ou_metrics is not None:
        trading_bounds = calculate_optimal_thresholds(ou_metrics)

        if trading_bounds is not None:
            hl_bars = ou_metrics["half_life"]
            
            return {
                "effective_date": effective_date, "ticker_a": ticker_a, "ticker_b": ticker_b,
                "mu": ou_metrics["mu"], "theta": ou_metrics["theta"], "sigma": ou_metrics["sigma"],
                "half_life_bars": hl_bars, "half_life_hours": (hl_bars * 5) / 60, 
                "long_entry": trading_bounds["long_entry"], "long_exit": trading_bounds["long_exit"],
                "short_entry": trading_bounds["short_entry"], "short_exit": trading_bounds["short_exit"],
            }
            
    return {
        "effective_date": effective_date, "ticker_a": ticker_a, "ticker_b": ticker_b,
        "mu": 0.0, "theta": 0.0, "sigma": 0.0, "half_life_bars": 0.0, "half_life_hours": 0.0, 
        "long_entry": -999999.0, "long_exit": 999999.0, "short_entry": 999999.0, "short_exit": -999999.0,
    }