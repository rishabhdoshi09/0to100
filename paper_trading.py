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
