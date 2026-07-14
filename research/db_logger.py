import sqlite3
import os

def get_db_connection():
    """Establishes a connection to the local SQLite database."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "outputs", "backtest_results.db")
    
    conn = sqlite3.connect(db_path)
    
    conn.execute("PRAGMA foreign_keys = 1")
    return conn

def initialize_database():
    """Creates the relational tables if they do not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT,
            lookback_bars INTEGER,
            step_bars INTEGER,
            kalman_delta REAL,
            kalman_obs_cov REAL,
            exit_target_pct REAL,
            transaction_cost REAL,
            leverage REAL,
            final_bot_equity REAL,
            bnh_benchmark REAL,
            outperformance_pct REAL,
            annualized_sharpe REAL,
            max_drawdown_pct REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pair_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            ticker_a TEXT,
            ticker_b TEXT,
            total_trades INTEGER,
            bot_final_equity REAL,
            bnh_final_equity REAL,
            outperformance_pct REAL,
            sharpe_ratio REAL,
            max_drawdown_pct REAL,
            FOREIGN KEY (run_id) REFERENCES portfolio_runs (run_id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def log_backtest_run(portfolio_record, pair_records):
    """Inserts the master record and all pair records in a single transaction."""
    initialize_database()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        columns = ', '.join(portfolio_record.keys())
        placeholders = ', '.join('?' * len(portfolio_record))
        
        cursor.execute(
            f"INSERT INTO portfolio_runs ({columns}) VALUES ({placeholders})",
            tuple(portfolio_record.values())
        )
        
        if len(pair_records) > 0:
            pair_columns = ', '.join(pair_records[0].keys())
            pair_placeholders = ', '.join('?' * len(pair_records[0]))
            
            cursor.executemany(
                f"INSERT INTO pair_results ({pair_columns}) VALUES ({pair_placeholders})",
                [tuple(record.values()) for record in pair_records]
            )
            
        conn.commit()
        print(f"\nRun {portfolio_record['run_id']} successfully saved to SQLite database.")
        
    except sqlite3.Error as e:
        print(f"\nDatabase Error: {e}")
        conn.rollback()
    finally:
        conn.close()

