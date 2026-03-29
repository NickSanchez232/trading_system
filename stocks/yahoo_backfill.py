import os
import time
import psycopg2
import yfinance as yf
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

def backfill_stock(symbol, start_date="2015-01-01"):
    print(f"  Fetching {symbol} from {start_date}...")

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=date.today().strftime("%Y-%m-%d"))

        if hist.empty:
            print(f"    No data returned for {symbol}")
            return []

        records = []
        for idx, row in hist.iterrows():
            records.append({
                "symbol": symbol,
                "date": idx.date().strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "vwap": None,
                "num_trades": None,
            })

        print(f"    Got {len(records)} days of data")
        return records

    except Exception as e:
        print(f"    Failed: {e}")
        return []
    
def save_backfill(records):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for r in records:
        cursor.execute(
            "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
            (r["symbol"],)
        )

        cursor.execute(
            """INSERT INTO daily_prices
                (symbol, date, open, high, low, close, volume, vwap, num_trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING""",
            (
                r["symbol"],
                r["date"],
                r["open"],
                r["high"],
                r["low"],
                r["close"],
                r["volume"],
                r["vwap"],
                r["num_trades"],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    return total_saved


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    grand_total = 0

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"[{i}/{len(WATCHLIST)}] Backfilling {symbol}...")
        records = backfill_stock(symbol)
        if records:
            saved = save_backfill(records)
            grand_total += saved
            print(f"    Saved {saved} records")
        time.sleep(2)

    print(f"\nBackfill complete! {grand_total} total records added.")
    