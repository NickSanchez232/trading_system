import os
import time
from datetime import date, timedelta
from dotenv import load_dotenv
from stocks.polygon_scraper import get_daily_prices, WATCHLIST, rate_limit_pause

import psycopg2

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("POLYGON_API_KEY")


def run_daily_scrape():
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

    print(f"Running daily scrape for {yesterday} to {today}...")

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    total_saved = 0

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"[{i}/{len(WATCHLIST)}] Fetching {symbol}...")
        prices = get_daily_prices(symbol, yesterday, today)

        cursor.execute(
            "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
            (symbol,)
        )

        for bar in prices:
            from datetime import datetime
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

        if i < len(WATCHLIST):
            rate_limit_pause()

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Done! Saved {total_saved} records.")


if __name__ == "__main__":
    run_daily_scrape()