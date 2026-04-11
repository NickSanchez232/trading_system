import os
import time
import requests
import psycopg2
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "User-Agent": "TradingSystem/0.1 (contact@example.com)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

CIK_CACHE = {}


def get_cik(symbol):
    if symbol in CIK_CACHE:
        return CIK_CACHE[symbol]

    time.sleep(0.5)
    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        response = SESSION.get(url, timeout=30)
        data = response.json()

        for key, info in data.items():
            CIK_CACHE[info.get("ticker", "").upper()] = str(info["cik_str"]).zfill(10)

        if symbol.upper() in CIK_CACHE:
            return CIK_CACHE[symbol.upper()]

    except Exception as e:
        print(f"    CIK lookup failed: {e}")

    return None


def get_buybacks(symbol, days_back=3650):
    print(f"  Fetching buybacks for {symbol}...")
    cik = get_cik(symbol)
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.2)

    try:
        response = SESSION.get(url, timeout=30)
        data = response.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])

        cutoff = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        buybacks = []

        for i, form in enumerate(forms):
            if form not in ["8-K", "8-K/A"]:
                continue

            filing_date = filing_dates[i] if i < len(filing_dates) else None
            if not filing_date or filing_date < cutoff:
                continue

            desc = descriptions[i].lower() if i < len(descriptions) and descriptions[i] else ""

            if any(word in desc for word in ["repurchase", "buyback", "buy back", "share repurchase"]):
                buybacks.append({
                    "symbol": symbol,
                    "filing_date": filing_date,
                    "form_type": form,
                    "description": descriptions[i] if i < len(descriptions) else "",
                })

        # Also check for common 8-K items related to buybacks
        # Item 8.01 is often used for buyback announcements
        if not buybacks:
            for i, form in enumerate(forms):
                if form not in ["8-K", "8-K/A"]:
                    continue

                filing_date = filing_dates[i] if i < len(filing_dates) else None
                if not filing_date or filing_date < cutoff:
                    continue

                desc = descriptions[i].lower() if i < len(descriptions) and descriptions[i] else ""

                if "item 8.01" in desc or "regulation fd" in desc:
                    buybacks.append({
                        "symbol": symbol,
                        "filing_date": filing_date,
                        "form_type": form,
                        "description": descriptions[i] if i < len(descriptions) else "",
                    })

        print(f"    Found {len(buybacks)} potential buyback filings")
        return buybacks

    except Exception as e:
        print(f"    Failed: {e}")
        return []


def calculate_buyback_signals(symbol, buybacks, price_dates):
    signals = []

    buyback_dates = set()
    for b in buybacks:
        buyback_dates.add(b["filing_date"])

    for _, row in price_dates.iterrows():
        current_date = str(row['date'])

        # Days since last buyback announcement
        past_buybacks = [d for d in buyback_dates if d <= current_date]
        days_since_buyback = None
        if past_buybacks:
            last_buyback = max(past_buybacks)
            days_since_buyback = (datetime.strptime(current_date, "%Y-%m-%d") - datetime.strptime(last_buyback, "%Y-%m-%d")).days

        # Buyback count in last 365 days
        one_year_ago = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        recent_buybacks = [d for d in buyback_dates if d >= one_year_ago and d <= current_date]
        buyback_count_1y = len(recent_buybacks)

        # Active buyback program
        has_active_buyback = days_since_buyback is not None and days_since_buyback <= 180

        signals.append({
            "symbol": symbol,
            "date": row['date'],
            "days_since_buyback": days_since_buyback,
            "buyback_count_1y": buyback_count_1y,
            "has_active_buyback": has_active_buyback,
        })

    return signals


def save_buyback_signals(all_signals):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for s in all_signals:
        try:
            cursor.execute(
                """INSERT INTO buyback_signals
                    (symbol, date, days_since_buyback, buyback_count_1y, has_active_buyback)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING""",
                (
                    s["symbol"],
                    s["date"],
                    s["days_since_buyback"],
                    s["buyback_count_1y"],
                    s["has_active_buyback"],
                )
            )
            total_saved += 1
        except Exception:
            conn.rollback()
            continue

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} buyback signal records to database.")


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST
    from sqlalchemy import create_engine
    import pandas as pd

    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)
    price_dates = pd.read_sql("SELECT DISTINCT symbol, date FROM daily_prices ORDER BY symbol, date", engine)
    engine.dispose()

    all_signals = []

    for symbol in WATCHLIST:
        buybacks = get_buybacks(symbol)
        symbol_dates = price_dates[price_dates['symbol'] == symbol]
        if len(symbol_dates) > 0:
            signals = calculate_buyback_signals(symbol, buybacks, symbol_dates)
            all_signals.extend(signals)
        time.sleep(0.5)

    save_buyback_signals(all_signals)