import os
import time
import requests
import psycopg2
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()


def get_stocktwits_sentiment(symbol):
    print(f"  Fetching StockTwits for {symbol}...")

    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"    StockTwits error: status {response.status_code}")
            return {
                "symbol": symbol,
                "bullish_count": 0,
                "bearish_count": 0,
                "total_messages": 0,
                "bull_bear_ratio": None,
                "sentiment_score": None,
            }

        data = response.json()
        messages = data.get("messages", [])

        bullish = 0
        bearish = 0
        total = len(messages)

        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment")
            if sentiment:
                if sentiment.get("basic") == "Bullish":
                    bullish += 1
                elif sentiment.get("basic") == "Bearish":
                    bearish += 1

        bull_bear_ratio = None
        if bearish > 0:
            bull_bear_ratio = bullish / bearish
        elif bullish > 0:
            bull_bear_ratio = 99.0

        sentiment_score = None
        if bullish + bearish > 0:
            sentiment_score = (bullish - bearish) / (bullish + bearish)

        print(f"    Messages: {total} | Bullish: {bullish} | Bearish: {bearish}")

        return {
            "symbol": symbol,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "total_messages": total,
            "bull_bear_ratio": bull_bear_ratio,
            "sentiment_score": sentiment_score,
        }

    except Exception as e:
        print(f"    StockTwits failed: {e}")
        return {
            "symbol": symbol,
            "bullish_count": 0,
            "bearish_count": 0,
            "total_messages": 0,
            "bull_bear_ratio": None,
            "sentiment_score": None,
        }


def save_stocktwits(all_data):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    today = date.today().strftime("%Y-%m-%d")
    total_saved = 0

    for d in all_data:
        try:
            cursor.execute(
                "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
                (d["symbol"],)
            )

            cursor.execute(
                """INSERT INTO stocktwits_sentiment
                    (symbol, date, bullish_count, bearish_count, total_messages,
                     bull_bear_ratio, sentiment_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING""",
                (
                    d["symbol"],
                    today,
                    d["bullish_count"],
                    d["bearish_count"],
                    d["total_messages"],
                    d["bull_bear_ratio"],
                    d["sentiment_score"],
                )
            )
            total_saved += 1
        except Exception:
            conn.rollback()
            continue

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} StockTwits records to database.")


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    all_data = []

    for symbol in WATCHLIST:
        result = get_stocktwits_sentiment(symbol)
        all_data.append(result)
        time.sleep(1)

    save_stocktwits(all_data)