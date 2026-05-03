import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "paper_trading.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        entry_date TEXT,
        entry_price REAL,
        quantity INTEGER,
        direction TEXT,
        status TEXT,
        exit_date TEXT,
        exit_price REAL,
        pnl REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS equity_curve (
        date TEXT PRIMARY KEY,
        equity REAL
    )''')
    conn.commit()
    conn.close()

def open_position(symbol, price, quantity, direction, date):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO positions (symbol, entry_date, entry_price, quantity, direction, status) VALUES (?,?,?,?,?,?)",
              (symbol, date, price, quantity, direction, 'open'))
    conn.commit()
    conn.close()

def close_position(pos_id, exit_price, exit_date):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT entry_price, quantity, direction FROM positions WHERE id=?", (pos_id,))
    row = c.fetchone()
    if row:
        entry_price, qty, direction = row
        if direction == 'BUY':
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty
        c.execute("UPDATE positions SET status='closed', exit_date=?, exit_price=?, pnl=? WHERE id=?", (exit_date, exit_price, pnl, pos_id))
    conn.commit()
    conn.close()

def get_open_positions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM positions WHERE status='open'", conn)
    conn.close()
    return df

def update_equity(date, total_equity):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO equity_curve (date, equity) VALUES (?,?)", (date, total_equity))
    conn.commit()
    conn.close()

def get_closed_positions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM positions WHERE status='closed' ORDER BY exit_date DESC", conn)
    conn.close()
    return df

def get_equity_curve():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT date, equity FROM equity_curve ORDER BY date ASC", conn)
    conn.close()
    return df

def get_trading_summary():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM positions WHERE status='closed'", conn)
    conn.close()
    if df.empty:
        return {"total_pnl": 0.0, "win_rate": 0.0, "num_trades": 0, "best_trade": 0.0, "worst_trade": 0.0}
    total_pnl = df['pnl'].sum()
    wins = (df['pnl'] > 0).sum()
    win_rate = (wins / len(df)) * 100 if len(df) > 0 else 0.0
    return {
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 1),
        "num_trades": len(df),
        "best_trade": round(df['pnl'].max(), 2),
        "worst_trade": round(df['pnl'].min(), 2),
    }
