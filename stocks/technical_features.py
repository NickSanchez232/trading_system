import os
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_price_data(symbol):
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    query = """
        SELECT symbol, date, open, high, low, close, volume, vwap
        FROM daily_prices
        WHERE symbol = %s
        ORDER BY date ASC
    """

    df = pd.read_sql(query, engine, params=(symbol,))
    engine.dispose()

    print(f"Loaded {len(df)} days of price data for {symbol}")
    return df

def calculate_technicals(df):
    # 1. RSI (14-day)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # 2. MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    # 3. Volume vs 30 day average
    df['volume_avg_30d'] = df['volume'].rolling(window=30).mean()
    df['volume_vs_avg'] = df['volume'] / df['volume_avg_30d']
    # 4. Price vs 52 week high and low
    df['high_52w'] = df['high'].rolling(window=252).max()
    df['low_52w'] = df['low'].rolling(window=252).min()
    df['price_vs_52w_high'] = (df['close'] - df['high_52w']) / df['high_52w']
    df['price_vs_52w_low'] = (df['close'] - df['low_52w']) / df['low_52w']
    # 5. Distance from 200 day moving average
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['dist_from_200dma'] = (df['close'] - df['ma_200']) / df['ma_200']
    # 6. Distance from 50 day moving average
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['dist_from_50dma'] = (df['close'] - df['ma_50']) / df['ma_50']
    # 7. Price momentum 1 week and 1 month
    df['momentum_1w'] = df['close'].pct_change(periods=5)
    df['momentum_1m'] = df['close'].pct_change(periods=21)
    # 8. Volume confirming price movement
    price_up = df['close'] > df['close'].shift(1)
    volume_up = df['volume'] > df['volume_avg_30d']
    price_down = df['close'] < df['close'].shift(1)
    volume_down = df['volume'] > df['volume_avg_30d']
    df['volume_confirms'] = ((price_up & volume_up) | (price_down & volume_down)).astype(int)
    # 9. Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    # 10. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    df['atr_percent'] = df['atr_14'] / df['close']
    # 11. OBV (On Balance Volume)
    obv_direction = np.where(df['close'] > df['close'].shift(1), df['volume'],
                    np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['obv'] = obv_direction.cumsum()
    df['obv_ma_20'] = df['obv'].rolling(window=20).mean()
    df['obv_trend'] = (df['obv'] - df['obv_ma_20']) / df['obv_ma_20']
    # 12. Support and Resistance levels
    df['rolling_high_20'] = df['high'].rolling(window=20).max()
    df['rolling_low_20'] = df['low'].rolling(window=20).min()
    df['near_resistance'] = (df['rolling_high_20'] - df['close']) / df['close']
    df['near_support'] = (df['close'] - df['rolling_low_20']) / df['close']

    return df

def save_technicals(df):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for _, row in df.iterrows():
        if pd.isna(row.get('rsi_14')):
            continue

        cursor.execute(
            """INSERT INTO technicals
                (symbol, date, rsi_14, macd, macd_signal, macd_histogram,
                 volume_vs_avg, price_vs_52w_high, price_vs_52w_low,
                 dist_from_200dma, dist_from_50dma, momentum_1w, momentum_1m,
                 volume_confirms, bb_position, atr_percent, obv_trend,
                 near_resistance, near_support)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING""",
            (
                row['symbol'],
                row['date'],
                row['rsi_14'],
                row['macd'],
                row['macd_signal'],
                row['macd_histogram'],
                row['volume_vs_avg'],
                row['price_vs_52w_high'],
                row['price_vs_52w_low'],
                row['dist_from_200dma'],
                row['dist_from_50dma'],
                row['momentum_1w'],
                row['momentum_1m'],
                row['volume_confirms'],
                row['bb_position'],
                row['atr_percent'],
                row['obv_trend'],
                row['near_resistance'],
                row['near_support'],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} technical records for {df['symbol'].iloc[0]}")

if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    for symbol in WATCHLIST:
        df = get_price_data(symbol)
        df = calculate_technicals(df)
        save_technicals(df)

    print("\nAll technicals calculated and saved!")