import os
import requests
import numpy as np
import polars as pl
from pykalman import KalmanFilter
from datetime import datetime, timedelta
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config # noqa: E402 # type: ignore
from research.math_engine import fit_ou_parameters, calculate_optimal_thresholds # noqa: E402 # type: ignore

class RecalibrationEngine:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.pairs_path = os.path.join(self.script_dir, "..", "outputs", "active_johansen_pairs.csv")
        self.output_path = os.path.join(self.script_dir, "..", "outputs", "live_ou_parameters.csv")
        
        self.tiingo_api_key = config.TIINGO_API_KEY
        
        self.lookback_days = int((config.LOOKBACK_BARS / 78) + 5)

    def fetch_tiingo_5m_data(self, ticker):
        """Fetches the required historical window from Tiingo."""
        print(f"Fetching recent data for {ticker}...")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        url = f"https://api.tiingo.com/iex/{ticker}/prices"
        params = {
            'startDate': start_date,
            'resampleFreq': '5min',
            'token': self.tiingo_api_key
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"[ERROR] Tiingo API failed for {ticker}: {response.text}")
            return None
            
        data = response.json()
        if not data:
            return None
            
        df = pl.DataFrame(data)
        df = df.rename({"date": "date", "close": ticker}).select(["date", ticker])
        df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ"))
        return df

    def calculate_live_kalman_state(self, df_a, df_b, ticker_a, ticker_b):        
        pair_df = df_a.join(df_b, on="date", how="inner").sort("date").drop_nulls()
        
        if len(pair_df) < 500: 
            return None, None, None, None

        log_a = np.log(pair_df[ticker_a].to_numpy())
        log_b = np.log(pair_df[ticker_b].to_numpy())
        
        obs_mat = np.vstack([log_b, np.ones(len(log_b))]).T
        obs_mat = np.expand_dims(obs_mat, axis=1)

        trans_cov = config.KALMAN_DELTA / (1 - config.KALMAN_DELTA) * np.eye(2)
        obs_cov = config.KALMAN_OBS_COV
        
        kf = KalmanFilter(
            n_dim_obs=1, n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            transition_covariance=trans_cov,
            observation_matrices=obs_mat,
            observation_covariance=obs_cov
        )
        
        state_means, _ = kf.filter(log_a)
        beta_raw = state_means[:, 0]
        alpha_raw = state_means[:, 1]

        beta = np.concatenate(([beta_raw[0]], beta_raw[:-1]))
        alpha = np.concatenate(([alpha_raw[0]], alpha_raw[:-1]))
        
        spread = log_a - (beta * log_b + alpha)
        
        final_beta = beta[-1]
        final_alpha = alpha[-1]
        
        return spread, pair_df["date"], final_beta, final_alpha

    def run_recalibration(self):
        print("Starting Portfolio Recalibration")

        active_pairs = pl.read_csv(self.pairs_path)
        new_parameters = []

        for row in active_pairs.iter_rows(named=True):
            t_a = row["ticker_a"]
            t_b = row["ticker_b"]
            
            df_a = self.fetch_tiingo_5m_data(t_a)
            df_b = self.fetch_tiingo_5m_data(t_b)
            
            if df_a is None or df_b is None:
                continue
                
            spread, dates, beta_T, alpha_T = self.calculate_live_kalman_state(df_a, df_b, t_a, t_b)
            
            if spread is None:
                continue
                
            if len(spread) > config.LOOKBACK_BARS:
                spread = spread[-config.LOOKBACK_BARS:]
                
            ou_metrics = fit_ou_parameters(spread) 
            
            if ou_metrics is not None:
                bounds = calculate_optimal_thresholds(ou_metrics)
                
                if bounds is not None:
                    print(f"[{t_a}/{t_b}] Beta: {beta_T:.4f} | Mu: {ou_metrics['mu']:.4f} | L_Entry: {bounds['long_entry']:.4f}")
                    
                    new_parameters.append({
                        "ticker_a": t_a,
                        "ticker_b": t_b,
                        "beta": beta_T,
                        "alpha": alpha_T,
                        "mu": ou_metrics["mu"],
                        "theta": ou_metrics["theta"],
                        "sigma": ou_metrics["sigma"],
                        "long_entry": bounds["long_entry"],
                        "long_exit": bounds["long_exit"],
                        "short_entry": bounds["short_entry"],
                        "short_exit": bounds["short_exit"]
                    })
                else:
                    print(f"[KILL SWITCH] {t_a}/{t_b} spread too tight to overcome friction.")
                    new_parameters.append({
                        "ticker_a": t_a, "ticker_b": t_b,
                        "beta": beta_T, "alpha": alpha_T,
                        "mu": 0.0, "theta": 0.0, "sigma": 0.0,
                        "long_entry": -999999.0, "long_exit": 999999.0,
                        "short_entry": 999999.0, "short_exit": -999999.0
                    })
            else:
                print(f"[KILL SWITCH] {t_a}/{t_b} lost mean reversion characteristics.")
                new_parameters.append({
                    "ticker_a": t_a, "ticker_b": t_b,
                    "beta": beta_T, "alpha": alpha_T,
                    "mu": 0.0, "theta": 0.0, "sigma": 0.0,
                    "long_entry": -999999.0, "long_exit": 999999.0,
                    "short_entry": 999999.0, "short_exit": -999999.0
                })
                
        if new_parameters:
            df_out = pl.DataFrame(new_parameters)
            df_out.write_csv(self.output_path)
            print(f"\nSuccessfully saved {len(df_out)} parameters to {self.output_path}")

if __name__ == "__main__":
    engine = RecalibrationEngine()
    engine.run_recalibration()