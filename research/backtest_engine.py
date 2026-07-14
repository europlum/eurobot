import os
import polars as pl
import numpy as np
from pykalman import KalmanFilter
from datetime import datetime, timezone
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config # noqa: E402 # type: ignore

def load_and_clean_data(filepath, price_col_name):
    df = pl.read_csv(filepath, try_parse_dates=True)
    
    df = df.fill_null(strategy="forward")
    
    df = df.with_columns([
        pl.col("close").pct_change().alias("ret_backward")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("ret_backward").abs() > 0.045)
        .then(None)
        .otherwise(pl.col("close"))
        .alias("clean_close")
    ])
    
    df = df.with_columns([
        pl.col("clean_close").forward_fill().alias(price_col_name)
    ])
    
    return df.select(["date", price_col_name])

def run_vectorized_backtest(ticker_a, ticker_b, pair_params_df):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_a = os.path.join(script_dir, "..", "data", f"{ticker_a}_5m.csv")
    file_b = os.path.join(script_dir, "..", "data", f"{ticker_b}_5m.csv")
    
    df_a = load_and_clean_data(file_a, "price_a")
    df_b = load_and_clean_data(file_b, "price_b")
    
    warmup_start_dt = datetime.strptime(config.SELECTION_START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(config.BACKTEST_END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    df_a = df_a.filter((pl.col("date") >= warmup_start_dt) & (pl.col("date") <= end_dt))
    df_b = df_b.filter((pl.col("date") >= warmup_start_dt) & (pl.col("date") <= end_dt))
    
    if len(df_a) == 0 or len(df_b) == 0:
        raise ValueError("No pricing data found in window.")

    valid_start = max(df_a["date"].min(), df_b["date"].min())
    valid_end = min(df_a["date"].max(), df_b["date"].max())
    
    df_a = df_a.filter((pl.col("date") >= valid_start) & (pl.col("date") <= valid_end))
    df_b = df_b.filter((pl.col("date") >= valid_start) & (pl.col("date") <= valid_end))
    
    aligned_df = df_a.join(df_b, on="date", how="outer").sort("date").fill_null(strategy="forward").drop_nulls()
    
    aligned_df = aligned_df.with_columns([
        pl.col("price_a").log().alias("log_a"),
        pl.col("price_b").log().alias("log_b")
    ])
    
    log_a_np = aligned_df["log_a"].to_numpy()
    log_b_np = aligned_df["log_b"].to_numpy()
    
    obs_mat = np.vstack([log_b_np, np.ones(len(log_b_np))]).T
    obs_mat = np.expand_dims(obs_mat, axis=1)
    trans_cov = config.KALMAN_DELTA / (1 - config.KALMAN_DELTA) * np.eye(2)
    
    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        transition_covariance=trans_cov,
        observation_matrices=obs_mat,
        observation_covariance=config.KALMAN_OBS_COV
    )
    
    state_means, _ = kf.filter(log_a_np)
    beta = state_means[:, 0]
    alpha = state_means[:, 1]
    
    aligned_df = aligned_df.with_columns([
        pl.Series("beta", beta).shift(1).fill_null(strategy="forward"),
        pl.Series("alpha", alpha).shift(1).fill_null(strategy="forward")
    ])
    
    aligned_df = aligned_df.with_columns(
        (pl.col("log_a") - (pl.col("beta") * pl.col("log_b") + pl.col("alpha"))).alias("spread")
    )
    
    backtest_start_dt = datetime.strptime(config.BACKTEST_START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    aligned_df = aligned_df.filter(pl.col("date") >= backtest_start_dt)
    
    pair_params_df = pair_params_df.rename({"effective_date": "date"}).sort("date")
    
    aligned_df = aligned_df.join_asof(
        pair_params_df.select(["date", "long_entry", "long_exit", "short_entry", "short_exit"]),
        on="date",
        strategy="backward"
    )
    
    aligned_df = aligned_df.drop_nulls(subset=["long_entry"])
    
    aligned_df = aligned_df.with_columns([
        pl.when(pl.col("spread") > pl.col("short_entry")).then(-1)  
        .when(pl.col("spread") < pl.col("long_entry")).then(1)      
        .when((pl.col("spread") >= pl.col("long_exit")) & (pl.col("spread") <= pl.col("short_exit"))).then(0)
        .otherwise(None)
        .alias("raw_signal")
    ])

    aligned_df = aligned_df.with_columns(pl.col("raw_signal").forward_fill().fill_null(0).alias("target_position"))
    aligned_df = aligned_df.with_columns(pl.col("target_position").shift(1).fill_null(0).alias("executed_position"))
    
    aligned_df = aligned_df.with_columns([
        (pl.col("price_a").log() - pl.col("price_a").log().shift(1)).fill_null(0).alias("ret_a"),
        (pl.col("price_b").log() - pl.col("price_b").log().shift(1)).fill_null(0).alias("ret_b")
    ])
    
    aligned_df = aligned_df.with_columns(
        (pl.col("ret_a") - (pl.col("beta") * pl.col("ret_b"))).alias("spread_return")
    )
    
    aligned_df = aligned_df.with_columns(
        (pl.col("executed_position") * pl.col("spread_return") * (config.LEVERAGE / (1.0 + pl.col("beta").abs()))).alias("gross_return")
    )
    
    aligned_df = aligned_df.with_columns(
        (pl.col("executed_position") - pl.col("executed_position").shift(1))
        .abs()
        .fill_null(0)
        .alias("trade_triggered")
    )
    
    bar_margin_cost = config.ANNUAL_MARGIN_RATE / (252 * 78)
    
    aligned_df = aligned_df.with_columns([
        (pl.col("trade_triggered") * config.TRANSACTION_COST * config.LEVERAGE).alias("execution_costs"),
        
        (pl.col("executed_position").abs() * bar_margin_cost * (config.LEVERAGE - 1)).alias("borrow_costs") 
    ])
    
    aligned_df = aligned_df.with_columns(
        (pl.col("gross_return") - pl.col("execution_costs") - pl.col("borrow_costs")).alias("net_return")
    )

    aligned_df = aligned_df.with_columns([
        (pl.col("net_return").exp().cum_prod()).alias("equity_curve")
    ])
    
    return aligned_df