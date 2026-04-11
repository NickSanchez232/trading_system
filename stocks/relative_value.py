import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Which stocks belong to which sector
SECTOR_GROUPS = {
    "Technology": ["NVDA", "AMD", "AAPL", "MSFT", "TSM", "AAOI", "TTD", "CRM", "INTC", "PLTR", "SMCI", "ENPH", "QQQ"],
    "Financial": ["JPM", "GS", "V", "SOFI", "COIN", "MARA", "RIOT", "UPST", "AFRM", "HOOD"],
    "Energy": ["XOM", "OXY", "DVN", "SLB"],
    "Healthcare": ["BSX", "PFE", "UNH", "JNJ"],
    "Consumer Cyclical": ["TSLA", "NKE", "LCID", "GME", "AMC", "CVNA", "MCD"],
    "Consumer Defensive": ["WMT", "PG", "KO"],
    "Industrials": ["DAL", "LMT"],
    "Materials": ["FCX", "NEM"],
}

# Reverse lookup - stock to sector
STOCK_TO_SECTOR = {}
for sector, stocks in SECTOR_GROUPS.items():
    for stock in stocks:
        STOCK_TO_SECTOR[stock] = sector


def get_price_data():
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    query = "SELECT symbol, date, close FROM daily_prices ORDER BY symbol, date"
    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"Loaded {len(df)} price records")
    return df


def calculate_relative_value(df, lookback=21):
    print(f"Calculating relative value signals with {lookback}-day lookback...")

    # Calculate returns for each stock
    all_features = []
    symbols = df['symbol'].unique()
    total = len(symbols)

    # Pre-calculate returns for all stocks
    returns_by_symbol = {}
    for symbol in symbols:
        stock_df = df[df['symbol'] == symbol].sort_values('date').copy()
        stock_df['return_21d'] = stock_df['close'].pct_change(periods=lookback)
        stock_df['return_5d'] = stock_df['close'].pct_change(periods=5)
        returns_by_symbol[symbol] = stock_df

    for idx, symbol in enumerate(symbols, 1):
        if idx % 10 == 0:
            print(f"  [{idx}/{total}] Processing {symbol}...")

        sector = STOCK_TO_SECTOR.get(symbol)
        if not sector:
            continue

        sector_peers = [s for s in SECTOR_GROUPS[sector] if s != symbol and s in returns_by_symbol]

        stock_df = returns_by_symbol[symbol]

        for _, row in stock_df.iterrows():
            current_date = row['date']
            stock_return_21d = row['return_21d']
            stock_return_5d = row['return_5d']

            if pd.isna(stock_return_21d):
                all_features.append({
                    "symbol": symbol,
                    "date": current_date,
                    "relative_value_21d": None,
                    "relative_value_5d": None,
                    "sector_divergence": None,
                    "peer_count": 0,
                })
                continue

            # Get peer returns for the same date
            peer_returns_21d = []
            peer_returns_5d = []
            for peer in sector_peers:
                peer_df = returns_by_symbol[peer]
                peer_row = peer_df[peer_df['date'] == current_date]
                if len(peer_row) > 0:
                    pr_21 = peer_row['return_21d'].values[0]
                    pr_5 = peer_row['return_5d'].values[0]
                    if not pd.isna(pr_21):
                        peer_returns_21d.append(pr_21)
                    if not pd.isna(pr_5):
                        peer_returns_5d.append(pr_5)

            # Calculate relative value
            relative_21d = None
            relative_5d = None
            sector_divergence = None

            if peer_returns_21d:
                sector_avg_21d = np.mean(peer_returns_21d)
                relative_21d = stock_return_21d - sector_avg_21d

            if peer_returns_5d and not pd.isna(stock_return_5d):
                sector_avg_5d = np.mean(peer_returns_5d)
                relative_5d = stock_return_5d - sector_avg_5d

            # Sector divergence - how far from peers in standard deviations
            if peer_returns_21d and len(peer_returns_21d) >= 2:
                sector_std = np.std(peer_returns_21d)
                if sector_std > 0:
                    sector_divergence = (stock_return_21d - sector_avg_21d) / sector_std

            all_features.append({
                "symbol": symbol,
                "date": current_date,
                "relative_value_21d": float(relative_21d) if relative_21d is not None else None,
                "relative_value_5d": float(relative_5d) if relative_5d is not None else None,
                "sector_divergence": float(sector_divergence) if sector_divergence is not None else None,
                "peer_count": len(peer_returns_21d),
            })

    print(f"Generated {len(all_features)} relative value records")
    return all_features


def save_relative_value(features):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for f in features:
        try:
            cursor.execute(
                """INSERT INTO relative_value
                    (symbol, date, relative_value_21d, relative_value_5d,
                     sector_divergence, peer_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING""",
                (
                    f["symbol"],
                    f["date"],
                    f["relative_value_21d"],
                    f["relative_value_5d"],
                    f["sector_divergence"],
                    f["peer_count"],
                )
            )
            total_saved += 1
        except Exception:
            conn.rollback()
            continue

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} relative value records to database.")


if __name__ == "__main__":
    df = get_price_data()
    features = calculate_relative_value(df)
    save_relative_value(features)