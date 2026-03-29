import os
import psycopg2
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_price_data():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)

    query = "SELECT symbol, date, close FROM daily_prices ORDER BY symbol, date"
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} price records")
    return df


def generate_labels(df, window_days=30):
    print(f"Generating labels with {window_days}-day window...")

    labels = []

    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        for i in range(len(stock_df)):
            signal_date = stock_df.loc[i, 'date']
            signal_price = stock_df.loc[i, 'close']

            # Find the price approximately 30 days later
            future_mask = stock_df['date'] >= signal_date + timedelta(days=window_days)
            future_rows = stock_df[future_mask]

            if future_rows.empty:
                continue

            outcome_row = future_rows.iloc[0]
            outcome_date = outcome_row['date']
            outcome_price = outcome_row['close']

            return_pct = (outcome_price - signal_price) / signal_price

            # Classify the outcome
            if return_pct >= 0.10:
                label = "strong_positive"
            elif return_pct <= -0.10:
                label = "strong_negative"
            elif abs(return_pct) <= 0.05:
                label = "neutral"
            else:
                label = "moderate"

            labels.append({
                "symbol": symbol,
                "signal_date": signal_date,
                "outcome_date": outcome_date,
                "price_at_signal": float(signal_price),
                "price_at_outcome": float(outcome_price),
                "return_pct": float(return_pct),
                "label": label,
            })

    print(f"Generated {len(labels)} labels")

    # Show distribution
    label_counts = {}
    for l in labels:
        label_counts[l['label']] = label_counts.get(l['label'], 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    return labels

def save_labels(labels):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for l in labels:
        cursor.execute(
            """INSERT INTO labels
                (symbol, signal_date, outcome_date, price_at_signal,
                 price_at_outcome, return_pct, label)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, signal_date) DO NOTHING""",
            (
                l["symbol"],
                l["signal_date"],
                l["outcome_date"],
                l["price_at_signal"],
                l["price_at_outcome"],
                l["return_pct"],
                l["label"],
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