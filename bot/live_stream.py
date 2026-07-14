import os
import asyncio
import polars as pl
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from datetime import datetime, timezone

def load_active_pairs():
    """
    Reads the active_johansen_pairs.csv and returns a list of tuples.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "..", "outputs", "active_johansen_pairs.csv")
    df = pl.read_csv(filepath)
    pairs = list(zip(df["ticker_a"], df["ticker_b"]))
    return pairs

class LiveStreamManager:
    
    STALE_THRESHOLD_SECONDS = 600 
    
    def __init__(self, api_key, secret_key, on_5m_close_callback):
        self.api_key = api_key
        self.secret_key = secret_key
        self.on_5m_close_callback = on_5m_close_callback
        
        self.active_pairs = load_active_pairs()
        self.unique_tickers = list(set([ticker for pair in self.active_pairs for ticker in pair]))
        
        self.latest_prices = {} 
        self.current_5m_interval = None
        self.stream = StockDataStream(self.api_key, self.secret_key, feed=DataFeed.IEX)
        
    async def handle_minute_bar(self, bar):
        symbol = bar.symbol
        close_price = bar.close
        bar_time = bar.timestamp
        
        interval_time = bar_time.replace(minute=(bar_time.minute // 5) * 5, second=0, microsecond=0)
        if self.current_5m_interval is None:
            self.current_5m_interval = interval_time
            
        self.latest_prices[symbol] = (close_price, bar_time)
        if interval_time > self.current_5m_interval:
            snapshot = self.build_snapshot()
            asyncio.create_task(self.on_5m_close_callback(self.current_5m_interval, snapshot, self.active_pairs))
            self.current_5m_interval = interval_time
    
    def build_snapshot(self):
        now = datetime.now(timezone.utc)
        snapshot = {}
        for symbol, (price, ts) in self.latest_prices.items():
            age = (now - ts).total_seconds()
            if age <= self.STALE_THRESHOLD_SECONDS:
                snapshot[symbol] = price
            else:
                print(f"[STALE] Excluding {symbol} — last update was {age:.0f}s ago")
        return snapshot
    
    def start(self):
        print(f"Starting live stream for: {self.unique_tickers}")
        self.stream.subscribe_bars(self.handle_minute_bar, *self.unique_tickers)
        self.stream.run()