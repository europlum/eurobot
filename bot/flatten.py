import os
import sys
from alpaca.trading.client import TradingClient

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
if root_dir not in sys.path: 
    sys.path.insert(0, root_dir)
    
import config # noqa: E402 # type: ignore

def flatten():
    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
    
    positions = client.get_all_positions()
    
    if not positions:
        print("No open positions found. Portfolio is already completely flat.")
        return

    print(f"Found {len(positions)} open positions")
    client.cancel_orders()
    
    try:
        client.close_all_positions(cancel_orders=True)
        print("All positions liquidated.")
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")

if __name__ == "__main__":
    flatten()