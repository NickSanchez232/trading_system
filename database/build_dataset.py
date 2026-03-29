import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def build_training_dataset():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)

    print("Loading labels...")
    labels = pd.read_sql("SELECT symbol, signal_date, return_pct, label FROM labels", conn)
    print(f"  {len(labels)} labels")

    print("Loading technicals...")
    technicals = pd.read_sql("SELECT * FROM technicals", conn)
    print(f"  {len(technicals)} technical records")

    print("Loading fundamentals...")
    fundamentals = pd.read_sql("SELECT * FROM fundamentals", conn)
    print(f"  {len(fundamentals)} fundamental records")

    print("Loading macro data...")
    macro = pd.read_sql("SELECT * FROM macro_data", conn)
    print(f"  {len(macro)} macro records")

    print("Loading stock signals...")
    stock_signals = pd.read_sql("SELECT * FROM stock_signals", conn)
    print(f"  {len(stock_signals)} stock signal records")

    print("Loading sentiment...")
    sentiment = pd.read_sql("SELECT * FROM sentiment", conn)
    print(f"  {len(sentiment)} sentiment records")

    conn.close()

    # Merge everything together on symbol and date
    print("\nMerging datasets...")

    # Start with labels as the base
    dataset = labels.copy()
    dataset = dataset.rename(columns={"signal_date": "date"})

    # Merge technicals
    tech_cols = [c for c in technicals.columns if c not in ['id', 'created_at']]
    dataset = dataset.merge(technicals[tech_cols], on=['symbol', 'date'], how='left')

    # Merge fundamentals - use the most recent fundamental data for each stock
    fund_cols = [c for c in fundamentals.columns if c not in ['id', 'created_at']]
    fundamentals_latest = fundamentals.sort_values('date').groupby('symbol').last().reset_index()
    fund_merge_cols = [c for c in fund_cols if c != 'date']
    dataset = dataset.merge(fundamentals_latest[fund_merge_cols], on='symbol', how='left')

    # Merge macro - use the closest macro data for each date
    macro_cols = [c for c in macro.columns if c not in ['id', 'created_at']]
    macro_latest = macro.sort_values('date').iloc[-1:]
    for col in macro_cols:
        if col != 'date':
            dataset[col] = macro_latest[col].values[0]

    # Merge stock signals
    signal_cols = [c for c in stock_signals.columns if c not in ['id', 'created_at']]
    signals_latest = stock_signals.sort_values('date').groupby('symbol').last().reset_index()
    signal_merge_cols = [c for c in signal_cols if c != 'date']
    dataset = dataset.merge(signals_latest[signal_merge_cols], on='symbol', how='left')

    # Merge sentiment
    sent_cols = [c for c in sentiment.columns if c not in ['id', 'created_at']]
    sentiment_latest = sentiment.sort_values('date').groupby('symbol').last().reset_index()
    sent_merge_cols = [c for c in sent_cols if c != 'date']
    dataset = dataset.merge(sentiment_latest[sent_merge_cols], on='symbol', how='left')

    print(f"\nFinal dataset: {len(dataset)} rows x {len(dataset.columns)} columns")
    print(f"\nColumns: {list(dataset.columns)}")

    # Show how many rows have complete data
    complete = dataset.dropna(subset=['rsi_14'])
    print(f"Rows with technical data: {len(complete)}")
    print(f"Rows missing technical data: {len(dataset) - len(complete)}")

    return dataset

def save_dataset(dataset):
    # Save to CSV for easy model training later
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, 'training_dataset.csv')
    dataset.to_csv(csv_path, index=False)
    print(f"\nDataset saved to {csv_path}")

    # Show label distribution in final dataset
    print("\nLabel distribution:")
    for label in dataset['label'].unique():
        count = len(dataset[dataset['label'] == label])
        print(f"  {label}: {count}")

    # Show sample row
    print("\nSample row:")
    print(dataset.iloc[0].to_string())


if __name__ == "__main__":
    dataset = build_training_dataset()
    save_dataset(dataset)