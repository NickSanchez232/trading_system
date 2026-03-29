import os
import time
import requests
import psycopg2
from psycopg2 import extras
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

def rate_limit_pause():
    print("Pausing for rate limit...")
    time.sleep(13)

WATCHLIST = [
    # Your current holdings
    "TSLA", "TSM", "NVDA", "AAOI", "TTD", "DAL", "CRM", "BSX",
    # Tech
    "AAPL", "MSFT", "AMD",
    # Finance
    "JPM", "GS", "V",
    # Energy / Oil & Gas
    "XOM", "OXY", "DVN", "SLB",
    # Healthcare
    "PFE", "UNH",
    # Retail
    "WMT", "NKE",
    # Mining / Materials
    "FCX", "NEM",
    # Defense
    "LMT",
    # High insider activity
    "INTC", "UBER", "PLTR", "SOFI", "COIN",
    # High volatility
    "MARA", "RIOT", "SMCI", "ENPH", "LCID",
    # High short interest
    "GME", "AMC", "CVNA", "UPST", "AFRM",
    # Heavy options flow
    "SPY", "QQQ", "IWM", "MSTR", "HOOD",
    # Stable blue chips
    "PG", "KO", "JNJ", "MCD", "T",
]

def get_daily_prices(symbol, from_date, to_date):
    endpoint = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": API_KEY,
    }
    response = requests.get(endpoint, params=params, timeout=30)
    data = response.json()
    results = data.get("results", [])

    print(f"Fetched {len(results)} days of data for {symbol}")
    return results

def fetch_all_stocks(from_date, to_date):
    all_data = {}
    total = len(WATCHLIST)

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"[{i}/{total}] Fetching {symbol}...")
        prices = get_daily_prices(symbol, from_date, to_date)
        all_data[symbol] = prices

        if i < total:
            rate_limit_pause()

    total_records = sum(len(v) for v in all_data.values())
    print(f"\nDone! Fetched {total_records} total records across {total} stocks.")
    return all_data

def save_to_database(all_data):
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for symbol, prices in all_data.items():
        # Make sure stock exists in master table
        cursor.execute(
            "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
            (symbol,)
        )

        for bar in prices:
            cursor.execute(
                """INSERT INTO daily_prices (symbol, date, open, high, low, close, volume, vwap, num_trades)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING""",
                (
                    symbol,
                    datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d'),
                    bar.get('o'),
                    bar.get('h'),
                    bar.get('l'),
                    bar.get('c'),
                    bar.get('v'),
                    bar.get('vw'),
                    bar.get('n'),
                )
            )
            total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} records to database.")

if __name__ == "__main__":
    data = fetch_all_stocks("2025-01-01", "2026-03-21")
    save_to_database(data)