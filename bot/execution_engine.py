import math
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import polars as pl
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config # noqa: E402 # type: ignore

class ExecutionEngine:
    def __init__(self):
        self.client = TradingClient(
            config.ALPACA_API_KEY, 
            config.ALPACA_SECRET_KEY, 
            paper=True
        )
        
        account = self.client.get_account()
        total_cash = float(account.portfolio_value)
        
        max_overnight_bp = total_cash * config.LEVERAGE
        safe_portfolio_limit = max_overnight_bp * config.MAX_ACCOUNT_UTILIZATION
        
        self.active_pairs_count = self.get_active_pair_count() 
        
        if self.active_pairs_count > 0:
            self.capital_per_pair = safe_portfolio_limit / self.active_pairs_count
        else:
            self.capital_per_pair = 0.0
            
        print(f"Total Equity: ${total_cash:,.2f} | Allocating ${self.capital_per_pair:,.2f} per pair across {self.active_pairs_count} pairs.")
    
    def get_active_pair_count(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, ".."))
        params_path = os.path.join(root_dir, "outputs", "live_ou_parameters.csv")
        
        try:
            df = pl.read_csv(params_path)
            valid_pairs = df.filter(pl.col("mu") > 0.0).height
            return valid_pairs
        except Exception:
            return 1 

    def get_tradable_asset(self, symbol):
        try:
            asset = self.client.get_asset(symbol)
            if not asset.tradable:
                return False, f"{symbol} is not tradable."
            if not asset.shortable:
                return False, f"{symbol} is not shortable."
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def calculate_order_quantities(self, price_a, price_b, hedge_ratio):
        spread_unit_cost = price_a + (abs(hedge_ratio * price_b))
        
        units = self.capital_per_pair / spread_unit_cost
        
        shares_a_raw = units
        shares_b_raw = units * hedge_ratio
        
        shares_a = math.floor(shares_a_raw)
        shares_b = math.floor(shares_b_raw)
        
        return shares_a, shares_b

    def execute_market_order(self, symbol, qty, side):
        if qty <= 0:
            return

        print(f"Routing Order: {side.name} {qty} shares of {symbol}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        self.client.submit_order(order_data=req)

    def route_orders(self, pending_orders, market_snapshot, ou_parameters):
        confirmed = []

        for order in pending_orders:
            ticker_a, ticker_b = order['pair']
            action = order['action']

            if action == 'FLAT':
                print(f"Liquidating positions for {ticker_a}/{ticker_b}")
                try:
                    self.client.close_position(ticker_a)
                    self.client.close_position(ticker_b)
                    confirmed.append(order)
                except Exception as e:
                    print(f"[WARNING] Could not close positions for {ticker_a}/{ticker_b}: {e}")
                continue

            price_a = market_snapshot.get(ticker_a)
            price_b = market_snapshot.get(ticker_b)
            hedge_ratio = ou_parameters[order['pair']]['beta']

            shares_a, shares_b = self.calculate_order_quantities(price_a, price_b, hedge_ratio)

            if shares_a == 0 or shares_b == 0:
                print(f"[WARNING] Calculated 0 shares for {ticker_a}/{ticker_b}. Skipping.")
                continue

            check_a, msg_a = self.get_tradable_asset(ticker_a)
            check_b, msg_b = self.get_tradable_asset(ticker_b)

            if not check_a or not check_b:
                print(f"[ABORT] Cannot trade {ticker_a}/{ticker_b}. Reason: {msg_a} | {msg_b}")
                continue

            try:
                if action == 'LONG_SPREAD':
                    self.execute_market_order(ticker_a, shares_a, OrderSide.BUY)
                    self.execute_market_order(ticker_b, shares_b, OrderSide.SELL)
                elif action == 'SHORT_SPREAD':
                    self.execute_market_order(ticker_a, shares_a, OrderSide.SELL)
                    self.execute_market_order(ticker_b, shares_b, OrderSide.BUY)

                confirmed.append(order)
            except Exception as e:
                print(f"[ERROR] Order execution failed for {ticker_a}/{ticker_b}: {e}")

        return confirmed