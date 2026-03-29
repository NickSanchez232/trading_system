import os
import time
import requests
import psycopg2
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_news_sentiment(symbol, company_name=None, days_back=7):
    print(f"  Fetching news for {symbol}...")

    search_term = symbol
    if company_name:
        search_term = company_name

    from_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = date.today().strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": search_term,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if data.get("status") != "ok":
            print(f"    News API error: {data.get('message')}")
            return {"symbol": symbol, "article_count": 0, "sentiment_score": None}

        articles = data.get("articles", [])
        print(f"    Found {len(articles)} articles")

        if not articles:
            return {"symbol": symbol, "article_count": 0, "sentiment_score": None}

        # Simple keyword-based sentiment scoring
        positive_words = [
            "surge", "soar", "jump", "rally", "gain", "beat", "record",
            "upgrade", "bullish", "growth", "profit", "strong", "buy",
            "outperform", "exceeded", "positive", "boom", "rise",
            "boost", "innovation", "breakthrough", "success",
            "dividend", "buyback", "expansion", "revenue", "earnings",
            "acquired", "partnership", "contract", "approved", "launch",
            "momentum", "recovery", "rebound", "optimistic", "confident",
            "raised guidance", "beat expectations", "all time high",
            "market share", "top pick", "overweight"
        ]

        negative_words = [
            "crash", "plunge", "drop", "fall", "miss", "loss", "decline",
            "downgrade", "bearish", "weak", "sell", "warning", "risk",
            "underperform", "negative", "fear", "cut", "down",
            "layoff", "recession", "debt", "lawsuit", "investigation",
            "bankruptcy", "fraud", "recall", "default", "shortage",
            "delay", "missed expectations", "lowered guidance",
            "overvalued", "bubble", "concern", "scrutiny", "penalty",
            "fine", "underweight", "sell off", "restructuring"
        ]

        total_score = 0
        scored_articles = 0

        for article in articles:
            title = (article.get("title") or "").lower()
            description = (article.get("description") or "").lower()
            text = title + " " + description

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count + neg_count > 0:
                article_score = (pos_count - neg_count) / (pos_count + neg_count)
                total_score += article_score
                scored_articles += 1

        sentiment_score = None
        if scored_articles > 0:
            sentiment_score = total_score / scored_articles

        print(f"    Sentiment score: {sentiment_score}")

        return {
            "symbol": symbol,
            "article_count": len(articles),
            "sentiment_score": sentiment_score,
        }

    except Exception as e:
        print(f"    News fetch failed: {e}")
        return {"symbol": symbol, "article_count": 0, "sentiment_score": None}
    
COMPANY_NAMES = {
    "TSLA": "Tesla",
    "TSM": "Taiwan Semiconductor",
    "NVDA": "Nvidia",
    "AAOI": "Applied Optoelectronics",
    "TTD": "Trade Desk",
    "DAL": "Delta Airlines",
    "CRM": "Salesforce",
    "BSX": "Boston Scientific",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMD": "AMD",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "V": "Visa",
    "XOM": "Exxon Mobil",
    "OXY": "Occidental Petroleum",
    "DVN": "Devon Energy",
    "SLB": "Schlumberger",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
    "WMT": "Walmart",
    "NKE": "Nike",
    "FCX": "Freeport McMoRan",
    "NEM": "Newmont Mining",
    "LMT": "Lockheed Martin",
}


def save_sentiment(all_sentiment):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    today = date.today().strftime("%Y-%m-%d")
    total_saved = 0

    for s in all_sentiment:
        cursor.execute(
            """INSERT INTO sentiment
                (symbol, date, news_sentiment_score, article_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING""",
            (
                s["symbol"],
                today,
                s["sentiment_score"],
                s["article_count"],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} sentiment records to database.")


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    all_sentiment = []

    for symbol in WATCHLIST:
        company_name = COMPANY_NAMES.get(symbol)
        result = get_news_sentiment(symbol, company_name)
        all_sentiment.append(result)
        time.sleep(1)

    save_sentiment(all_sentiment)