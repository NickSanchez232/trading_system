import os
import psycopg2
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


def get_price_data():
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    query = "SELECT symbol, date, close FROM daily_prices ORDER BY symbol, date"
    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"Loaded {len(df)} price records")
    return df


def generate_labels(df, configs=None):
    if configs is None:
        configs = [
            {"window_days": 14, "strong_threshold": 0.05, "neutral_band": 0.025, "suffix": "_2w"},
            {"window_days": 30, "strong_threshold": 0.10, "neutral_band": 0.05, "suffix": "_30d"},
        ]

    all_labels = []

    for config in configs:
        window = config["window_days"]
        strong = config["strong_threshold"]
        neutral = config["neutral_band"]
        suffix = config["suffix"]

        print(f"Generating labels: {window}-day window, {strong:.0%} threshold...")

        for symbol in df['symbol'].unique():
            stock_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

            for i in range(len(stock_df)):
                signal_date = stock_df.loc[i, 'date']
                signal_price = stock_df.loc[i, 'close']

                future_mask = stock_df['date'] >= signal_date + timedelta(days=window)
                future_rows = stock_df[future_mask]

                if future_rows.empty:
                    continue

                outcome_row = future_rows.iloc[0]
                outcome_date = outcome_row['date']
                outcome_price = outcome_row['close']

                return_pct = (outcome_price - signal_price) / signal_price

                # Track what happens during the trade window
                window_data = stock_df[
                    (stock_df['date'] > signal_date) &
                    (stock_df['date'] <= outcome_row['date'])
                ]

                if len(window_data) > 0:
                    max_price = window_data['close'].max()
                    min_price = window_data['close'].min()
                    max_gain = (max_price - signal_price) / signal_price
                    max_drawdown = (min_price - signal_price) / signal_price
                else:
                    max_gain = 0
                    max_drawdown = 0

                if return_pct >= strong:
                    label = "strong_positive"
                elif return_pct <= -strong:
                    label = "strong_negative"
                elif abs(return_pct) <= neutral:
                    label = "neutral"
                else:
                    label = "moderate"

                all_labels.append({
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "outcome_date": outcome_date,
                    "price_at_signal": float(signal_price),
                    "price_at_outcome": float(outcome_price),
                    "return_pct": float(return_pct),
                    "max_gain": float(max_gain),
                    "max_drawdown": float(max_drawdown),
                    "label": label,
                    "timeframe": suffix,
                })

    print(f"\nGenerated {len(all_labels)} total labels")

    for config in configs:
        suffix = config["suffix"]
        tf_labels = [l for l in all_labels if l["timeframe"] == suffix]
        print(f"\n{suffix} distribution:")
        label_counts = {}
        for l in tf_labels:
            label_counts[l['label']] = label_counts.get(l['label'], 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

    return all_labels


def save_labels(labels):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for l in labels:
        cursor.execute(
            """INSERT INTO labels
                (symbol, signal_date, outcome_date, price_at_signal,
                 price_at_outcome, return_pct, max_gain, max_drawdown, label, timeframe)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, signal_date, timeframe) DO NOTHING""",
            (
                l["symbol"],
                l["signal_date"],
                l["outcome_date"],
                l["price_at_signal"],
                l["price_at_outcome"],
                l["return_pct"],
                l["max_gain"],
                l["max_drawdown"],
                l["label"],
                l["timeframe"],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} labels to database.")


if __name__ == "__main__":
    df = get_price_data()
    labels = generate_labels(df)
    save_labels(labels)