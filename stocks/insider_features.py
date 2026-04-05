import os
import psycopg2
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


def get_insider_trades():
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    query = "SELECT symbol, filing_date, trade_date, insider_name, insider_title, trade_type, shares, price_per_share, total_value, shares_owned_after FROM insider_trades ORDER BY symbol, trade_date"
    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"Loaded {len(df)} insider trades")
    return df


def get_price_dates():
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    query = "SELECT DISTINCT symbol, date FROM daily_prices ORDER BY symbol, date"
    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"Loaded {len(df)} price date records")
    return df


def calculate_insider_features(price_dates, insider_trades):
    print("Calculating insider features...")

    all_features = []
    symbols = price_dates['symbol'].unique()
    total = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        if idx % 10 == 0:
            print(f"  [{idx}/{total}] Processing {symbol}...")

        symbol_dates = price_dates[price_dates['symbol'] == symbol].sort_values('date')
        symbol_trades = insider_trades[insider_trades['symbol'] == symbol]

        if symbol_trades.empty:
            for _, row in symbol_dates.iterrows():
                all_features.append({
                    "symbol": symbol,
                    "date": row['date'],
                    "ceo_bought_recently": False,
                    "cfo_bought_recently": False,
                    "multiple_insiders_buying": False,
                    "insider_buy_dollar_amt": 0,
                    "buy_sell_ratio_90d": None,
                    "insider_ownership_pct": None,
                    "days_since_last_buy": None,
                    "insider_buy_score": 0,
                    "buy_size_vs_avg": 0,
                    "consecutive_months_buying": 0,
                })
            continue

        symbol_trades = symbol_trades.copy()
        symbol_trades['trade_date'] = pd.to_datetime(symbol_trades['trade_date']).dt.date

        all_symbol_buys = symbol_trades[symbol_trades['trade_type'] == 'buy']['total_value']
        avg_buy_size = all_symbol_buys.mean() if len(all_symbol_buys) > 0 else 0

        for _, row in symbol_dates.iterrows():
            current_date = row['date']
            lookback_90 = current_date - timedelta(days=90)
            lookback_30 = current_date - timedelta(days=30)

            recent_90 = symbol_trades[
                (symbol_trades['trade_date'] >= lookback_90) &
                (symbol_trades['trade_date'] <= current_date)
            ]

            recent_30 = symbol_trades[
                (symbol_trades['trade_date'] >= lookback_30) &
                (symbol_trades['trade_date'] <= current_date)
            ]

            # 1. CEO bought recently
            ceo_buys = recent_30[
                (recent_30['insider_title'].str.contains('CEO|Chief Executive', case=False, na=False)) &
                (recent_30['trade_type'] == 'buy')
            ]
            ceo_bought = len(ceo_buys) > 0

            # 2. CFO bought recently
            cfo_buys = recent_30[
                (recent_30['insider_title'].str.contains('CFO|Chief Financial', case=False, na=False)) &
                (recent_30['trade_type'] == 'buy')
            ]
            cfo_bought = len(cfo_buys) > 0

            # 3. Multiple insiders buying simultaneously
            unique_buyers_30 = recent_30[recent_30['trade_type'] == 'buy']['insider_name'].nunique()
            multiple_buying = unique_buyers_30 >= 3

            # 4. Dollar amount of insider purchases last 90 days
            buy_dollar = recent_90[recent_90['trade_type'] == 'buy']['total_value'].sum()

            # 5. Buy vs sell ratio last 90 days
            buys_90 = len(recent_90[recent_90['trade_type'] == 'buy'])
            sells_90 = len(recent_90[recent_90['trade_type'] == 'sell'])
            if sells_90 > 0:
                buy_sell_ratio = buys_90 / sells_90
            elif buys_90 > 0:
                buy_sell_ratio = 99.0
            else:
                buy_sell_ratio = None

            # 6. Insider ownership percentage
            insider_ownership = None

            # 7. Days since last insider buy
            all_buys = symbol_trades[
                (symbol_trades['trade_type'] == 'buy') &
                (symbol_trades['trade_date'] <= current_date)
            ]
            if len(all_buys) > 0:
                last_buy_date = all_buys['trade_date'].max()
                days_since = (current_date - last_buy_date).days
            else:
                days_since = None

            # 8. Weighted insider buy score
            insider_score = 0
            for _, trade in recent_30[recent_30['trade_type'] == 'buy'].iterrows():
                title = str(trade.get('insider_title', '')).upper()
                if 'CEO' in title or 'CHIEF EXECUTIVE' in title:
                    insider_score += 3
                elif 'CFO' in title or 'CHIEF FINANCIAL' in title:
                    insider_score += 2
                elif 'VP' in title or 'VICE PRESIDENT' in title:
                    insider_score += 1.5
                else:
                    insider_score += 1

            # 9. Buy size vs average
            recent_buy_total = recent_30[recent_30['trade_type'] == 'buy']['total_value'].sum()
            buy_size_vs_avg = float(recent_buy_total / avg_buy_size) if avg_buy_size > 0 else 0

            # 10. Consecutive months of insider buying
            consecutive_months = 0
            for months_back in range(1, 7):
                month_start = current_date - timedelta(days=30 * months_back)
                month_end = current_date - timedelta(days=30 * (months_back - 1))
                month_buys = symbol_trades[
                    (symbol_trades['trade_type'] == 'buy') &
                    (symbol_trades['trade_date'] >= month_start) &
                    (symbol_trades['trade_date'] <= month_end)
                ]
                if len(month_buys) > 0:
                    consecutive_months += 1
                else:
                    break

            all_features.append({
                "symbol": symbol,
                "date": current_date,
                "ceo_bought_recently": ceo_bought,
                "cfo_bought_recently": cfo_bought,
                "multiple_insiders_buying": multiple_buying,
                "insider_buy_dollar_amt": float(buy_dollar),
                "buy_sell_ratio_90d": buy_sell_ratio,
                "insider_ownership_pct": insider_ownership,
                "days_since_last_buy": days_since,
                "insider_buy_score": insider_score,
                "buy_size_vs_avg": buy_size_vs_avg,
                "consecutive_months_buying": consecutive_months,
            })

    print(f"Generated {len(all_features)} insider feature records")
    return all_features


def save_insider_features(features):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for f in features:
        try:
            cursor.execute(
                """INSERT INTO insider_signals
                    (symbol, date, ceo_bought_recently, cfo_bought_recently,
                     multiple_insiders_buying, insider_buy_dollar_amt,
                     buy_sell_ratio_90d, insider_ownership_pct, days_since_last_buy,
                     insider_buy_score, buy_size_vs_avg, consecutive_months_buying)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING""",
                (
                    f["symbol"],
                    f["date"],
                    f["ceo_bought_recently"],
                    f["cfo_bought_recently"],
                    f["multiple_insiders_buying"],
                    f["insider_buy_dollar_amt"],
                    f["buy_sell_ratio_90d"],
                    f["insider_ownership_pct"],
                    f["days_since_last_buy"],
                    f["insider_buy_score"],
                    f["buy_size_vs_avg"],
                    f["consecutive_months_buying"],
                )
            )
            total_saved += 1
        except Exception:
            conn.rollback()
            continue

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} insider feature records to database.")


if __name__ == "__main__":
    insider_trades = get_insider_trades()
    price_dates = get_price_dates()
    features = calculate_insider_features(price_dates, insider_trades)
    save_insider_features(features)