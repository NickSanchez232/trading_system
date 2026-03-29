import os
import time
import requests
import psycopg2
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_congress_trades(symbol, days_back=365):
    # Senate Stock Watcher data from GitHub
    senate_url = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/data/aggregate/all_transactions.json"
    # House Stock Watcher data from GitHub
    house_url = "https://raw.githubusercontent.com/timothycarambat/house-stock-watcher-data/main/data/all_transactions.json"

    cutoff = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_symbol_trades = []

    # Try Senate data
    try:
        print(f"  Fetching Senate trades...")
        response = requests.get(senate_url, timeout=60)
        if response.status_code == 200:
            senate_trades = response.json()
            for trade in senate_trades:
                ticker = trade.get("ticker", "")
                trade_date = trade.get("transaction_date", "")
                if ticker == symbol and trade_date >= cutoff:
                    all_symbol_trades.append({
                        "source": "senate",
                        "representative": f"{trade.get('first_name', '')} {trade.get('last_name', '')}",
                        "trade_date": trade_date,
                        "trade_type": trade.get("type", ""),
                        "amount": trade.get("amount", ""),
                    })
    except Exception as e:
        print(f"    Senate data failed: {e}")

    # Try House data
    try:
        print(f"  Fetching House trades...")
        response = requests.get(house_url, timeout=60)
        if response.status_code == 200:
            house_trades = response.json()
            for trade in house_trades:
                ticker = trade.get("ticker", "")
                trade_date = trade.get("transaction_date", "")
                if ticker == symbol and trade_date >= cutoff:
                    all_symbol_trades.append({
                        "source": "house",
                        "representative": trade.get("representative", ""),
                        "trade_date": trade_date,
                        "trade_type": trade.get("type", ""),
                        "amount": trade.get("amount", ""),
                    })
    except Exception as e:
        print(f"    House data failed: {e}")

    buys = sum(1 for t in all_symbol_trades if "purchase" in t["trade_type"].lower())
    sells = sum(1 for t in all_symbol_trades if "sale" in t["trade_type"].lower())

    return {
        "symbol": symbol,
        "total_trades": len(all_symbol_trades),
        "congress_buys": buys,
        "congress_sells": sells,
    }


def save_congress_trades(all_congress_data):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    today = date.today().strftime("%Y-%m-%d")
    total_saved = 0

    for data in all_congress_data:
        cursor.execute(
            "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
            (data["symbol"],)
        )

        cursor.execute(
            """INSERT INTO congress_trades
                (symbol, date, total_trades, congress_buys, congress_sells)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING""",
            (
                data["symbol"],
                today,
                data["total_trades"],
                data["congress_buys"],
                data["congress_sells"],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} congress trade records to database.")


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    # Download both datasets once
    print("Downloading congressional trading data...")

    senate_trades = []
    house_trades = []

    try:
        senate_url = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/data/aggregate/all_transactions.json"
        response = requests.get(senate_url, timeout=60)
        if response.status_code == 200:
            senate_trades = response.json()
            print(f"  Senate: {len(senate_trades)} total trades downloaded")
    except Exception as e:
        print(f"  Senate download failed: {e}")

    try:
        house_url = "https://raw.githubusercontent.com/timothycarambat/house-stock-watcher-data/main/data/all_transactions.json"
        response = requests.get(house_url, timeout=60)
        if response.status_code == 200:
            house_trades = response.json()
            print(f"  House: {len(house_trades)} total trades downloaded")
    except Exception as e:
        print(f"  House download failed: {e}")

    cutoff = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    all_congress_data = []

    for symbol in WATCHLIST:
        senate_matches = [t for t in senate_trades if t.get("ticker") == symbol and t.get("transaction_date", "") >= cutoff]
        house_matches = [t for t in house_trades if t.get("ticker") == symbol and t.get("transaction_date", "") >= cutoff]
        combined = senate_matches + house_matches

        buys = sum(1 for t in combined if "purchase" in t.get("type", "").lower())
        sells = sum(1 for t in combined if "sale" in t.get("type", "").lower())

        print(f"  {symbol}: {len(combined)} trades ({buys} buys, {sells} sells)")

        all_congress_data.append({
            "symbol": symbol,
            "total_trades": len(combined),
            "congress_buys": buys,
            "congress_sells": sells,
        })

    save_congress_trades(all_congress_data)