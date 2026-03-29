import os
import time
import sys
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# Add the stocks folder to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stocks'))

from polygon_scraper import get_daily_prices, WATCHLIST, rate_limit_pause
from sec_edgar_scraper import get_insider_trades, save_insider_trades
from technical_features import get_price_data, calculate_technicals, save_technicals
from fundamental_features import get_fundamentals, calculate_sector_avg_pe, save_fundamentals
from macro_features import (get_vix_data, get_sp500_data, get_fed_rate,
                            get_fear_greed_index, get_sector_momentum,
                            get_credit_spreads, get_short_interest,
                            get_analyst_ratings, save_macro_data,
                            save_stock_signals, STOCK_SECTOR_MAP)
from sentiment_features import get_news_sentiment, save_sentiment, COMPANY_NAMES

import psycopg2
from datetime import datetime

DB_URL = os.getenv("DATABASE_URL")


def run_price_scrape():
    print("=" * 50)
    print("STEP 1: Daily Price Data")
    print("=" * 50)
    yesterday = (date.today() - timedelta(days=3)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

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
    print(f"Prices done! Saved {total_saved} records.\n")


def run_insider_scrape():
    print("=" * 50)
    print("STEP 2: Insider Trades")
    print("=" * 50)
    all_trades = []
    for symbol in WATCHLIST:
        trades = get_insider_trades(symbol, days_back=30)
        all_trades.extend(trades)
    save_insider_trades(all_trades)
    print(f"Insider trades done!\n")


def run_technicals():
    print("=" * 50)
    print("STEP 3: Technical Indicators")
    print("=" * 50)
    for symbol in WATCHLIST:
        df = get_price_data(symbol)
        df = calculate_technicals(df)
        save_technicals(df)
    print(f"Technicals done!\n")


def run_fundamentals():
    print("=" * 50)
    print("STEP 4: Fundamentals")
    print("=" * 50)
    all_fundamentals = []
    for symbol in WATCHLIST:
        f = get_fundamentals(symbol)
        all_fundamentals.append(f)
        time.sleep(1)
    all_fundamentals = calculate_sector_avg_pe(all_fundamentals)
    save_fundamentals(all_fundamentals)
    print(f"Fundamentals done!\n")


def run_macro():
    print("=" * 50)
    print("STEP 5: Macro & Stock Signals")
    print("=" * 50)
    vix_df = get_vix_data()
    sp500_df = get_sp500_data()
    fed_rate, fed_direction = get_fed_rate()
    fg_score, fg_rating = get_fear_greed_index()
    credit_spread, credit_direction = get_credit_spreads()
    sector_momentum = get_sector_momentum()

    save_macro_data(sp500_df, vix_df, fed_rate, fed_direction, fg_score, fg_rating, credit_spread, credit_direction)

    for symbol in WATCHLIST:
        short_data = get_short_interest(symbol)
        analyst_data = get_analyst_ratings(symbol)
        sector = STOCK_SECTOR_MAP.get(symbol)
        sector_vs_sp500 = None
        if sector and sector in sector_momentum:
            sector_vs_sp500 = float(sector_momentum[sector]["vs_sp500"])
        save_stock_signals(symbol, sector_vs_sp500, short_data, analyst_data)
        time.sleep(1)
    print(f"Macro & stock signals done!\n")


def run_sentiment():
    print("=" * 50)
    print("STEP 6: News Sentiment")
    print("=" * 50)
    all_sentiment = []
    for symbol in WATCHLIST:
        company_name = COMPANY_NAMES.get(symbol)
        result = get_news_sentiment(symbol, company_name)
        all_sentiment.append(result)
        time.sleep(1)
    save_sentiment(all_sentiment)
    print(f"Sentiment done!\n")


if __name__ == "__main__":
    print("=" * 50)
    print(f"FULL DATA COLLECTION - {date.today()}")
    print("=" * 50)

    run_price_scrape()
    run_insider_scrape()
    run_technicals()
    run_fundamentals()
    run_macro()
    run_sentiment()

    print("=" * 50)
    print("ALL DONE! Full pipeline complete.")
    print("=" * 50)