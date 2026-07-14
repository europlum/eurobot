import os
import polars as pl
import matplotlib.pyplot as plt
import datetime
import numpy as np
from backtest_engine import run_vectorized_backtest
from db_logger import log_backtest_run
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config # noqa: E402 # type: ignore

run_timestamp = datetime.datetime.now()
RUN_ID = f"RUN_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"

script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "..", "outputs", "portfolio_model_parameters.csv")

plots_dir = os.path.join(script_dir, "..", "outputs", "plots", RUN_ID)
os.makedirs(plots_dir, exist_ok=True) 

print(f"System Mode: {config.SYSTEM_MODE}")
print(f"Run ID initialized: {RUN_ID}")
print(f"Plots will be saved to: {plots_dir}\n")

portfolio_df = pl.read_csv(params_path, try_parse_dates=True)
unique_pairs = portfolio_df.select(["ticker_a", "ticker_b"]).unique()
print(f"Loaded time-series parameters for {len(unique_pairs)} pair(s).\n")

all_portfolio_data = []
all_pair_records = []

for row in unique_pairs.iter_rows(named=True):
    ticker_a = row["ticker_a"]
    ticker_b = row["ticker_b"]
    
    print(f"Initiating Backtest: {ticker_a} & {ticker_b}")
    pair_params_df = portfolio_df.filter(
        (pl.col("ticker_a") == ticker_a) & (pl.col("ticker_b") == ticker_b)
    )
    
    try:
        results_df = run_vectorized_backtest(
            ticker_a=ticker_a, 
            ticker_b=ticker_b, 
            pair_params_df=pair_params_df, 
        )
        
        dates = results_df["date"].to_numpy()
        bot_equity = results_df["equity_curve"].to_numpy()
        price_a = results_df["price_a"].to_numpy()
        price_b = results_df["price_b"].to_numpy()
        
        bnh_a = price_a / price_a[0]
        bnh_b = price_b / price_b[0]
        bnh_portfolio = (bnh_a + bnh_b) / 2.0
        
        final_bot_eq = bot_equity[-1]
        final_bnh_eq = bnh_portfolio[-1]
        
        total_trades = results_df["trade_triggered"].sum() / 2.0 
        
        print("--- Pair Results ---")
        print(f"  Total Trades Executed: {total_trades}")
        print(f"  Bot Final Equity: {final_bot_eq:.4f}x")
        print(f"  B&H Final Equity: {final_bnh_eq:.4f}x")
        print("-" * 30 + "\n")
        
        plt.figure(figsize=(10, 5))
        plt.plot(dates, bot_equity, label=f"Bot Strategy ({ticker_a}/{ticker_b})", color='blue', linewidth=2)
        plt.plot(dates, bnh_portfolio, label="50/50 B&H Benchmark", color='black', linewidth=1.5, linestyle='--')
        plt.axhline(y=1.0, color='red', linestyle=':', alpha=0.8)
        plt.title(f"Individual Pair: {ticker_a} & {ticker_b}")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pair_plot_path = os.path.join(plots_dir, f"{ticker_a}_{ticker_b}.png")
        plt.savefig(pair_plot_path, dpi=150)
        plt.close()
        
        pair_returns = results_df["net_return"].to_numpy()
        bars_per_year = 78 * 252
        
        pair_daily_returns = (
            results_df
            .with_columns(pl.col("date").dt.date().alias("trade_date"))
            .group_by("trade_date")
            .agg(pl.col("net_return").sum())
        )["net_return"].to_numpy()
        
        if np.std(pair_daily_returns) != 0:
            pair_sharpe = (np.mean(pair_daily_returns) / np.std(pair_daily_returns)) * np.sqrt(252)
            
        pair_cum_max = np.maximum.accumulate(bot_equity)
        pair_drawdowns = (bot_equity - pair_cum_max) / pair_cum_max
        pair_mdd = np.min(pair_drawdowns) * 100
        
        pair_record = {
            "run_id": RUN_ID,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "total_trades": int(total_trades),
            "bot_final_equity": round(final_bot_eq, 4),
            "bnh_final_equity": round(final_bnh_eq, 4),
            "outperformance_pct": round((final_bot_eq - final_bnh_eq) * 100, 2),
            "sharpe_ratio": round(pair_sharpe, 3),
            "max_drawdown_pct": round(pair_mdd, 2)
        }
        all_pair_records.append(pair_record)
        
        pair_returns_df = results_df.select(["date", "net_return", "ret_a", "ret_b"])
        all_portfolio_data.append(pair_returns_df)
        
    except Exception as e:
        print(f"Error running backtest for {ticker_a}/{ticker_b}: {e}")

if len(all_portfolio_data) > 0:
    print("\n" + "="*40)
    print("AGGREGATING PORTFOLIO-LEVEL PERFORMANCE")
    print("="*40)
    
    master_df = pl.concat(all_portfolio_data)
    
    agg_portfolio_df = master_df.group_by("date").agg([
        pl.col("net_return").mean().alias("port_bot_return"),
        ((pl.col("ret_a").mean() + pl.col("ret_b").mean()) / 2.0).alias("port_bnh_return")
    ]).sort("date")
    
    spy_path = os.path.join(script_dir, "..", "data", "SPY_5m.csv")
    has_spy = os.path.exists(spy_path)
    
    if has_spy:
        spy_df = pl.read_csv(spy_path, try_parse_dates=True)
        spy_df = spy_df.with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1)).fill_null(0).alias("spy_return")
        )
        
        agg_portfolio_df = agg_portfolio_df.join(
            spy_df.select(["date", "spy_return"]), 
            on="date", 
            how="left"
        ).fill_null(0) 
        
        agg_portfolio_df = agg_portfolio_df.with_columns([
            (pl.col("port_bot_return").exp().cum_prod()).alias("port_bot_equity"),
            (pl.col("port_bnh_return").exp().cum_prod()).alias("port_bnh_equity"),
            (pl.col("spy_return").exp().cum_prod()).alias("spy_equity")
        ])
        
    dates = agg_portfolio_df["date"].to_numpy()
    bot_equity = agg_portfolio_df["port_bot_equity"].to_numpy()
    bnh_equity = agg_portfolio_df["port_bnh_equity"].to_numpy()
    
    final_port_bot = bot_equity[-1]
    final_port_bnh = bnh_equity[-1]
    
    print(f"MASTER BOT EQUITY:           {final_port_bot:.4f}x")
    print(f"MASTER BUY & HOLD EQUITY:    {final_port_bnh:.4f}x")
    
    if has_spy:
        spy_equity = agg_portfolio_df["spy_equity"].to_numpy()
        final_spy_eq = spy_equity[-1]
        print(f"S&P 500 (SPY) EQUITY:        {final_spy_eq:.4f}x")
    
    outperformance_bnh = (final_port_bot - final_port_bnh) * 100
    if outperformance_bnh > 0:
        print(f"Bot OUTPERFORMED Pair Buy & Hold by {outperformance_bnh:.2f}%")
    else:
        print(f"Bot UNDERPERFORMED Pair Buy & Hold by {abs(outperformance_bnh):.2f}%")
    
    outperformance_spy = (final_port_bot - final_spy_eq) * 100
    if outperformance_spy > 0:
        print(f"Bot OUTPERFORMED S&P 500 by {outperformance_spy:.2f}%")
    else:
        print(f"Bot UNDERPERFORMED S&P 500 by {abs(outperformance_spy):.2f}%")
        
    plt.figure(figsize=(14, 7))
    plt.plot(dates, bot_equity, label=f"Aggregated Bot Portfolio ({final_port_bot:.2f}x)", color='blue', linewidth=2.5)
    plt.plot(dates, bnh_equity, label=f"Equally Weighted Pair B&H ({final_port_bnh:.2f}x)", color='black', linewidth=2, linestyle='-')
    
    if has_spy:
        plt.plot(dates, spy_equity, label=f"S&P 500 Benchmark ({final_spy_eq:.2f}x)", color='green', linewidth=2, linestyle=':')
    
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8)
    
    plt.title("MASTER PORTFOLIO: Total Strategy Performance vs. Benchmarks")
    plt.xlabel("Date (Intraday)")
    plt.ylabel("Cumulative Return Multiplier")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.figure(figsize=(14, 7))
    plt.plot(dates, bot_equity, label=f"Aggregated Bot Portfolio ({final_port_bot:.2f}x)", color='blue', linewidth=2.5)
    plt.plot(dates, bnh_equity, label=f"Equally Weighted Pair B&H ({final_port_bnh:.2f}x)", color='black', linewidth=2, linestyle='-')
    
    if has_spy:
        plt.plot(dates, spy_equity, label=f"S&P 500 Benchmark ({final_spy_eq:.2f}x)", color='green', linewidth=2, linestyle=':')
        
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8)
    
    plt.title(f"MASTER PORTFOLIO [{RUN_ID}]")
    plt.xlabel("Date (Intraday)")
    plt.ylabel("Cumulative Return Multiplier")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    master_plot_path = os.path.join(plots_dir, "00_MASTER_PORTFOLIO.png")
    plt.savefig(master_plot_path, dpi=300)
    # plt.show()
    plt.close()
else:
    print("No backtest data generated.")
    

if len(all_portfolio_data) > 0:
    port_returns = agg_portfolio_df["port_bot_return"].to_numpy()
    
    daily_portfolio = (
        agg_portfolio_df
        .with_columns(pl.col("date").dt.date().alias("trade_date"))
        .group_by("trade_date")
        .agg(pl.col("port_bot_return").sum())
    )["port_bot_return"].to_numpy()

    if np.std(daily_portfolio) != 0:
        annualized_sharpe = (np.mean(daily_portfolio) / np.std(daily_portfolio)) * np.sqrt(252)
        
    cumulative_max = np.maximum.accumulate(bot_equity)
    drawdowns = (bot_equity - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns) * 100 
     
    portfolio_record = {
        "run_id": RUN_ID,
        "timestamp": run_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "lookback_bars": config.LOOKBACK_BARS,
        "step_bars": config.STEP_BARS,
        "kalman_delta": config.KALMAN_DELTA,
        "kalman_obs_cov": config.KALMAN_OBS_COV,
        "exit_target_pct": config.EXIT_TARGET_PCT,
        "transaction_cost": config.TRANSACTION_COST,
        "leverage": config.LEVERAGE,
        "final_bot_equity": round(final_port_bot, 4),
        "bnh_benchmark": round(final_port_bnh, 4),
        "outperformance_pct": round(outperformance_bnh, 2),
        "annualized_sharpe": round(annualized_sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2)
    }
    
    log_backtest_run(portfolio_record, all_pair_records)