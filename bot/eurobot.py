import os
import sys
import math
import polars as pl
import signal
from live_stream import LiveStreamManager
from execution_engine import ExecutionEngine

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config # noqa: E402 # type: ignore

shutdown_requested = False

def shutdown(signum, frame):
    print("Shutdown Requested")
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGINT, shutdown)

class EuroBotOrchestrator:
    def __init__(self):
        self.params_path = os.path.join(root_dir, "outputs", "live_ou_parameters.csv")
        self.ou_parameters = self.load_parameters()
        self.executor = ExecutionEngine()
        
        self.current_positions = {} 
        
        self.reconcile_positions_with_broker()

    def reconcile_positions_with_broker(self):
        print("\nSyncing memory with Alpaca...")
        try:
            open_positions = self.executor.client.get_all_positions()
            broker_map = {p.symbol: float(p.qty) for p in open_positions}
            
            for pair_key in self.ou_parameters.keys():
                ticker_a, ticker_b = pair_key
                
                qty_a = broker_map.get(ticker_a, 0)
                qty_b = broker_map.get(ticker_b, 0)
                
                has_a = qty_a != 0
                has_b = qty_b != 0
                
                if has_a and has_b:
                    if qty_a > 0:
                        self.current_positions[pair_key] = 1
                        print(f"  LONG SPREAD on {ticker_a}/{ticker_b}")
                    else:
                        self.current_positions[pair_key] = -1
                        print(f"  SHORT SPREAD on {ticker_a}/{ticker_b}")
                        
                elif has_a or has_b:
                    orphan = ticker_a if has_a else ticker_b
                    print(f"  [WARNING] Orphaned leg detected: {orphan} in pair {ticker_a}/{ticker_b}. Closing.")
                    try:
                        self.executor.client.close_position(orphan)
                    except Exception as e:
                        print(f"  [ERROR] Could not close orphan {orphan}: {e}")
                    self.current_positions[pair_key] = 0
                    
                else:
                    self.current_positions[pair_key] = 0
                        
        except Exception as e:
            print(f"[ERROR] Could not sync with broker: {e}. Starting flat.")
            for pair_key in self.ou_parameters.keys():
                self.current_positions[pair_key] = 0

    def load_parameters(self):
        print("Loading live OU parameters from /outputs...")
        df = pl.read_csv(self.params_path)
        
        params_dict = {}
        for row in df.iter_rows(named=True):
            pair_key = (row['ticker_a'], row['ticker_b'])
            params_dict[pair_key] = row
        return params_dict
    
    async def evaluate_portfolio(self, interval_time, market_snapshot, active_pairs):
        global shutdown_requested
        print(f"\nEvaluating strategy for {interval_time.strftime('%H:%M')}...")
        pending_orders = []
        
        if shutdown_requested:
            print("Exiting main loop")
            sys.exit(0)

        for ticker_a, ticker_b in active_pairs:
            pair_key = (ticker_a, ticker_b)
            price_a = market_snapshot.get(ticker_a)
            price_b = market_snapshot.get(ticker_b)
            
            if not price_a or not price_b:
                continue 
                
            if pair_key not in self.ou_parameters:
                continue

            params = self.ou_parameters[pair_key]
            
            if params['mu'] == 0.0:
                continue
                
            beta = params['beta']   
            alpha = params['alpha'] 
            
            long_entry = params['long_entry']
            long_exit = params['long_exit']
            short_entry = params['short_entry']
            short_exit = params['short_exit']

            current_spread = math.log(price_a) - (beta * math.log(price_b) + alpha)
            
            print(f"{ticker_a}/{ticker_b} | Spread: {current_spread:.5f} | L_Ent: {long_entry:.5f} | S_Ent: {short_entry:.5f}")

            current_pos = self.current_positions.get(pair_key, 0)

            # ENTRY LOGIC
            if current_pos == 0:
                if current_spread <= long_entry:
                    print(f"*** SIGNAL: LONG SPREAD {ticker_a}/{ticker_b} ***")
                    pending_orders.append({'pair': pair_key, 'action': 'LONG_SPREAD', 'new_pos': 1})
                    
                elif current_spread >= short_entry:
                    print(f"*** SIGNAL: SHORT SPREAD {ticker_a}/{ticker_b} ***")
                    pending_orders.append({'pair': pair_key, 'action': 'SHORT_SPREAD', 'new_pos': -1})

            # EXIT LOGIC
            elif current_pos == 1 and current_spread >= long_exit:
                print(f"*** SIGNAL: EXIT LONG {ticker_a}/{ticker_b} ***")
                pending_orders.append({'pair': pair_key, 'action': 'FLAT', 'new_pos': 0})
                
            elif current_pos == -1 and current_spread <= short_exit:
                print(f"*** SIGNAL: EXIT SHORT {ticker_a}/{ticker_b} ***")
                pending_orders.append({'pair': pair_key, 'action': 'FLAT', 'new_pos': 0})

        if pending_orders:
            print(f"Sending {len(pending_orders)} orders to Execution Engine...")
            confirmed = self.executor.route_orders(pending_orders, market_snapshot, self.ou_parameters)
            for order in confirmed:
                self.current_positions[order['pair']] = order['new_pos']
        else:
            print("No actionable signals this interval")


if __name__ == "__main__":

    API_KEY = config.ALPACA_API_KEY
    SECRET_KEY = config.ALPACA_SECRET_KEY

    bot = EuroBotOrchestrator()
    
    streamer = LiveStreamManager(
        api_key=API_KEY, 
        secret_key=SECRET_KEY, 
        on_5m_close_callback=bot.evaluate_portfolio
    )
    
    streamer.start()