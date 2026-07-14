import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

LOOKBACK_BARS = 5000          # Window size for OU parameter fitting
STEP_BARS = 156              # Frequency of parameter recalculation

TRANSACTION_COST = 0.0002    # Assumed slippage + fee per leg (5 bps)
EXIT_TARGET_PCT = 0.8       # Percentage of the spread width captured before exiting

KALMAN_DELTA = 1e-7          # Transition covariance multiplier (controls beta drift speed)
KALMAN_OBS_COV = 1e-5        # Observation covariance (noise filter)

MAX_HALF_LIFE_BARS = 500     # Maximum allowed half-life before discarding the pair

LEVERAGE = 2.0              # Capital allocation
ANNUAL_MARGIN_RATE = 0.08    # Assumes Alpaca charges 8% annually for borrowed capital
MAX_ACCOUNT_UTILIZATION = 0.60

SYSTEM_MODE = "P" 

# 1. RESEARCH MODE
if SYSTEM_MODE == "R": 
    SELECTION_START_DATE = "2021-01-04"  
    SELECTION_END_DATE = "2023-12-31"    
    
    BACKTEST_START_DATE = "2024-01-01"   
    BACKTEST_END_DATE = "2026-12-31"     
    
    PAIRS_OUTPUT_FILE = "research_johansen_pairs.csv"

# 2. PRODUCTION MODE
elif SYSTEM_MODE == "P":
    SELECTION_START_DATE = "2021-01-04"  
    SELECTION_END_DATE = "2030-01-01"    
    
    BACKTEST_START_DATE = "2026-06-26"   
    BACKTEST_END_DATE = "2026-06-26"     
    
    PAIRS_OUTPUT_FILE = "active_johansen_pairs.csv"