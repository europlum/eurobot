import os
import time
from datetime import datetime
import polars as pl
import pandas as pd
import financedatabase as fd
import requests
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config # noqa: E402 # type: ignore

def apply_corporate_actions(intraday_df, ticker, api_key):
    """
    Fetches historical split data from Tiingo's Daily endpoint and 
    retroactively adjusts the 5-minute intraday prices and volume.
    """
    headers = {'Content-Type': 'application/json', 'Authorization': f"Token {api_key}"}
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=2020-01-01"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"      Warning: Could not fetch split data for {ticker}. Status: {response.status_code}")
            return intraday_df
            
        daily_data = pd.DataFrame(response.json())
        
        if 'splitFactor' in daily_data.columns:
            splits = daily_data[daily_data['splitFactor'] != 1.0]
            
            for index, row in splits.iterrows():
                split_date = pd.to_datetime(row['date'], utc=True)
                split_factor = row['splitFactor']
                
                print(f"      -> Applying {split_factor}-for-1 split on {split_date.date()} for {ticker}")
                
                mask = intraday_df.index < split_date
                
                for col in ['open', 'high', 'low', 'close']:
                    if col in intraday_df.columns:
                        intraday_df.loc[mask, col] = intraday_df.loc[mask, col] / split_factor
                
                if 'volume' in intraday_df.columns:
                    intraday_df.loc[mask, 'volume'] = intraday_df.loc[mask, 'volume'] * split_factor
                    
    except Exception as e:
        print(f"      Error adjusting splits for {ticker}: {e}")
        
    return intraday_df


def download_historical_intraday_chunks(tickers, start_date="2021-01-01", end_date=None, max_downloads = 2):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    date_bins = pd.date_range(start=start_date, end=end_date, freq='5MS').strftime('%Y-%m-%d').tolist()
    
    if date_bins[-1] != end_date:
        date_bins.append(end_date)

    download_count = 0
    
    for ticker in tickers:
        if download_count >= max_downloads:
            print(f"\nLimit reached: Downloaded {max_downloads} tickers.")
            break
        
        file_path = f"data/{ticker}_5m.csv"
        
        if os.path.exists(file_path):
            print(f"Data for {ticker} already exists. Skipping...")
            continue

        print(f"Starting chunked download for {ticker}...")
        ticker_chunks = []
        
        for i in range(len(date_bins) - 1):
            chunk_start = date_bins[i]
            chunk_end = date_bins[i+1]
            print(f"   Fetching {ticker}: {chunk_start} to {chunk_end}")
            
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f"Token {config.TIINGO_API_KEY}"
                }
                
                url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={chunk_start}&endDate={chunk_end}&resampleFreq=5min&columns=open,high,low,close,volume"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data: 
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        ticker_chunks.append(df)
                else:
                    print(f"   API Error {response.status_code} for {chunk_start}")
                    
                time.sleep(5) 
                
            except Exception as e:
                print(f"   Error fetching {ticker} chunk {chunk_start}: {e}")
                
        if ticker_chunks:
            master_df = pd.concat(ticker_chunks)
            master_df = master_df[~master_df.index.duplicated(keep='first')]
            master_df.sort_index(inplace=True)
            
            if master_df.index.tz is None:
                master_df.index = master_df.index.tz_localize('UTC')
            else:
                master_df.index = master_df.index.tz_convert('UTC')
            
            master_df = apply_corporate_actions(master_df, ticker, config.TIINGO_API_KEY)
            master_df.to_csv(file_path)
            print(f"Successfully saved {len(master_df)} adjusted rows for {ticker} to {file_path}\n")
            
            time.sleep(1.5)
        
        download_count += 1


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    master_df = pl.read_csv("data/ticker_master.csv").with_columns(
        pl.col("endDate").str.to_date("%Y-%m-%d", strict=False)
    )

    equities = fd.Equities()
    sector_df = pl.from_pandas(equities.select(), include_index=True).select([
        pl.col("symbol").alias("ticker"),
        pl.col("sector"),
        pl.col("industry")  
    ])

    master_df = master_df.join(sector_df, on="ticker", how="left")

    backtest_start_date = datetime.strptime("2020-01-01", "%Y-%m-%d").date()
    
    TARGET_TICKERS = []
    
    valid_tickers_unordered = master_df.filter(
        (pl.col("assetType") == "Stock") &
        (pl.col("priceCurrency") == "USD") &
        (
            pl.col("endDate").is_null() | 
            (pl.col("endDate") >= backtest_start_date)
        ) &
        (pl.col("ticker").is_in(TARGET_TICKERS))
    )["ticker"].to_list() 

    tickers_to_download = [ticker for ticker in TARGET_TICKERS if ticker in valid_tickers_unordered]
    
    print(f"Found {len(tickers_to_download)} symbols for data processing.")
    
    if tickers_to_download:
        print(f"Starting price download sequence for: {tickers_to_download}")
        download_historical_intraday_chunks(tickers_to_download)