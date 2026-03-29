import os
import time
import requests
import psycopg2
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

def get_vix_data(days_back=365):
    print("Fetching VIX data...")
    vix = yf.Ticker("^VIX")
    hist = vix.history(period=f"{days_back}d")

    if hist.empty:
        print("  No VIX data returned")
        return None

    df = hist[['Close']].reset_index()
    df.columns = ['date', 'vix_level']
    df['date'] = pd.to_datetime(df['date']).dt.date

    # VIX direction — is it rising or falling over last 5 days
    df['vix_ma_5'] = df['vix_level'].rolling(window=5).mean()
    df['vix_direction'] = df['vix_level'] - df['vix_ma_5']

    print(f"  Loaded {len(df)} days of VIX data")
    print(f"  Current VIX: {df['vix_level'].iloc[-1]:.2f}")
    return df

def get_sp500_data(days_back=365):
    print("Fetching S&P 500 data...")
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(period=f"{days_back}d")

    if hist.empty:
        print("  No S&P 500 data returned")
        return None

    df = hist[['Close', 'Volume']].reset_index()
    df.columns = ['date', 'sp500_level', 'sp500_volume']
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Market regime based on 50 and 200 day moving averages
    df['sp500_ma_50'] = df['sp500_level'].rolling(window=50).mean()
    df['sp500_ma_200'] = df['sp500_level'].rolling(window=200).mean()

    def classify_regime(row):
        if pd.isna(row['sp500_ma_50']) or pd.isna(row['sp500_ma_200']):
            return None
        if row['sp500_level'] > row['sp500_ma_50'] > row['sp500_ma_200']:
            return 'bull'
        elif row['sp500_level'] < row['sp500_ma_50'] < row['sp500_ma_200']:
            return 'bear'
        else:
            return 'sideways'

    df['market_regime'] = df.apply(classify_regime, axis=1)

    print(f"  Loaded {len(df)} days of S&P 500 data")
    print(f"  Current regime: {df['market_regime'].iloc[-1]}")
    return df

def get_fed_rate():
    print("Fetching Fed rate data...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "FEDFUNDS",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "sort_order": "desc",
        "limit": 12,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            print("  No Fed rate data returned")
            return None, None

        current_rate = float(observations[0]["value"])
        previous_rate = float(observations[1]["value"]) if len(observations) > 1 else current_rate

        if current_rate > previous_rate:
            rate_direction = "rising"
        elif current_rate < previous_rate:
            rate_direction = "falling"
        else:
            rate_direction = "stable"

        print(f"  Current Fed rate: {current_rate}%")
        print(f"  Direction: {rate_direction}")
        return current_rate, rate_direction

    except Exception as e:
        print(f"  Fed rate fetch failed: {e}")
        return None, None
    
def get_fear_greed_index():
    print("Fetching Fear & Greed Index...")

    urls = [
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        "https://production.dataviz.cnn.io/index/fearandgreed/current",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                continue

            data = response.json()

            # Try different response structures
            if "fear_and_greed" in data:
                score = data["fear_and_greed"].get("score")
                rating = data["fear_and_greed"].get("rating")
            elif "score" in data:
                score = data["score"]
                rating = data.get("rating")
            else:
                continue

            if score is not None:
                print(f"  Fear & Greed Score: {float(score):.0f}")
                print(f"  Rating: {rating}")
                return float(score), rating

        except Exception:
            continue

    # Fallback: calculate from VIX as a rough estimate
    print("  CNN API failed. Using VIX-based estimate...")
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            vix_level = hist['Close'].iloc[-1]
            # Rough conversion: low VIX = greed, high VIX = fear
            score = max(0, min(100, 100 - (vix_level * 2.5)))
            if score > 75:
                rating = "Extreme Greed"
            elif score > 55:
                rating = "Greed"
            elif score > 45:
                rating = "Neutral"
            elif score > 25:
                rating = "Fear"
            else:
                rating = "Extreme Fear"
            print(f"  VIX-based Fear & Greed estimate: {score:.0f}")
            print(f"  Rating: {rating}")
            return score, rating
    except Exception as e:
        print(f"  Fallback also failed: {e}")

    return None, None
    
def get_sector_momentum(days_back=30):
    print("Fetching sector momentum vs S&P 500...")
    sectors = {
        "XLK": "Technology",
        "XLF": "Financial",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLY": "Consumer Cyclical",
        "XLP": "Consumer Defensive",
        "XLI": "Industrials",
        "XLB": "Materials",
    }

    sp500 = yf.Ticker("^GSPC")
    sp500_hist = sp500.history(period=f"{days_back}d")
    if sp500_hist.empty:
        print("  No S&P 500 data")
        return {}

    sp500_return = (sp500_hist['Close'].iloc[-1] - sp500_hist['Close'].iloc[0]) / sp500_hist['Close'].iloc[0]

    sector_momentum = {}

    for etf, name in sectors.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period=f"{days_back}d")
            if hist.empty:
                continue
            sector_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
            vs_sp500 = sector_return - sp500_return
            sector_momentum[name] = {
                "sector_return": sector_return,
                "sp500_return": sp500_return,
                "vs_sp500": vs_sp500,
            }
            print(f"  {name}: {sector_return:.2%} vs S&P {sp500_return:.2%} ({vs_sp500:+.2%})")
        except Exception:
            continue

    return sector_momentum

def get_credit_spreads():
    print("Fetching credit/bond spread data...")
    url = "https://api.stlouisfed.org/fred/series/observations"

    # ICE BofA US High Yield Option-Adjusted Spread
    params = {
        "series_id": "BAMLH0A0HYM2",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "sort_order": "desc",
        "limit": 30,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            print("  No credit spread data returned")
            return None, None

        # Filter out any non-numeric values
        values = []
        for obs in observations:
            try:
                values.append(float(obs["value"]))
            except ValueError:
                continue

        if not values:
            return None, None

        current_spread = values[0]
        avg_spread = sum(values) / len(values)

        if current_spread > avg_spread:
            spread_direction = "widening"
        elif current_spread < avg_spread:
            spread_direction = "tightening"
        else:
            spread_direction = "stable"

        print(f"  Current spread: {current_spread:.2f}%")
        print(f"  30-day avg: {avg_spread:.2f}%")
        print(f"  Direction: {spread_direction}")
        return current_spread, spread_direction

    except Exception as e:
        print(f"  Credit spread fetch failed: {e}")
        return None, 

def get_short_interest(symbol):
    print(f"  Fetching short interest for {symbol}...")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        short_pct = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")

        if short_pct is not None:
            print(f"    Short % of float: {short_pct:.2%}")
            print(f"    Short ratio: {short_ratio}")
        else:
            print(f"    No short interest data available")

        return {
            "symbol": symbol,
            "short_pct_of_float": short_pct,
            "short_ratio": short_ratio,
        }

    except Exception as e:
        print(f"    Short interest fetch failed: {e}")
        return {
            "symbol": symbol,
            "short_pct_of_float": None,
            "short_ratio": None,
        }
    
def get_analyst_ratings(symbol):
    print(f"  Fetching analyst ratings for {symbol}...")
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations

        if recommendations is None or len(recommendations) == 0:
            print(f"    No analyst data available")
            return {
                "symbol": symbol,
                "total_buy": None,
                "total_sell": None,
                "total_hold": None,
                "buy_sell_ratio": None,
            }

        latest = recommendations.iloc[-1]

        strong_buy = latest.get("strongBuy", 0)
        buy = latest.get("buy", 0)
        hold = latest.get("hold", 0)
        sell = latest.get("sell", 0)
        strong_sell = latest.get("strongSell", 0)

        total_buy = strong_buy + buy
        total_sell = sell + strong_sell
        total_hold = hold

        buy_sell_ratio = None
        if total_sell > 0:
            buy_sell_ratio = total_buy / total_sell
        elif total_buy > 0:
            buy_sell_ratio = 99.0

        print(f"    Buy: {total_buy} | Hold: {total_hold} | Sell: {total_sell}")
        print(f"    Buy/Sell ratio: {buy_sell_ratio}")

        return {
            "symbol": symbol,
            "total_buy": total_buy,
            "total_sell": total_sell,
            "total_hold": total_hold,
            "buy_sell_ratio": buy_sell_ratio,
        }

    except Exception as e:
        print(f"    Analyst ratings fetch failed: {e}")
        return {
            "symbol": symbol,
            "total_buy": None,
            "total_sell": None,
            "total_hold": None,
            "buy_sell_ratio": None,
        }
    
def save_macro_data(sp500_df, vix_df, fed_rate, fed_direction, fg_score, fg_rating, credit_spread, credit_direction):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    today = date.today().strftime("%Y-%m-%d")
    sp500_level = None
    market_regime = None
    vix_level = None
    vix_dir = None

    if sp500_df is not None and len(sp500_df) > 0:
        sp500_level = float(sp500_df['sp500_level'].iloc[-1])
        market_regime = sp500_df['market_regime'].iloc[-1]

    if vix_df is not None and len(vix_df) > 0:
        vix_level = float(vix_df['vix_level'].iloc[-1])
        vix_dir = float(vix_df['vix_direction'].iloc[-1])

    cursor.execute(
        """INSERT INTO macro_data
            (date, sp500_level, market_regime, vix_level, vix_direction,
             fed_rate, fed_rate_direction, fear_greed_score, fear_greed_rating,
             credit_spread, credit_spread_direction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) DO NOTHING""",
        (
            today, sp500_level, market_regime, vix_level, vix_dir,
            fed_rate, fed_direction, fg_score, fg_rating,
            credit_spread, credit_direction,
        )
    )

    conn.commit()
    cursor.close()
    conn.close()
    print("Saved macro data to database.")


def save_stock_signals(symbol, sector_vs_sp500, short_data, analyst_data):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    today = date.today().strftime("%Y-%m-%d")

    def to_python(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
        
    cursor.execute(
        """INSERT INTO stock_signals
            (symbol, date, sector_momentum_vs_sp500, short_pct_of_float,
             short_ratio, analyst_total_buy, analyst_total_sell,
             analyst_total_hold, analyst_buy_sell_ratio)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, date) DO NOTHING""",
        (
            symbol,
            today,
            to_python(sector_vs_sp500),
            to_python(short_data["short_pct_of_float"]),
            to_python(short_data["short_ratio"]),
            to_python(analyst_data["total_buy"]),
            to_python(analyst_data["total_sell"]),
            to_python(analyst_data["total_hold"]),
            to_python(analyst_data["buy_sell_ratio"]),
        )
    )

    conn.commit()
    cursor.close()
    conn.close()


# Map stocks to their sectors for momentum lookup
STOCK_SECTOR_MAP = {
    "TSLA": "Consumer Cyclical",
    "TSM": "Technology",
    "NVDA": "Technology",
    "AAOI": "Technology",
    "TTD": "Technology",
    "DAL": "Industrials",
    "CRM": "Technology",
    "BSX": "Healthcare",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "AMD": "Technology",
    "JPM": "Financial",
    "GS": "Financial",
    "V": "Financial",
    "XOM": "Energy",
    "OXY": "Energy",
    "DVN": "Energy",
    "SLB": "Energy",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    "WMT": "Consumer Defensive",
    "NKE": "Consumer Cyclical",
    "FCX": "Materials",
    "NEM": "Materials",
    "LMT": "Industrials",
}


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    # Fetch market-wide data
    vix_df = get_vix_data()
    sp500_df = get_sp500_data()
    fed_rate, fed_direction = get_fed_rate()
    fg_score, fg_rating = get_fear_greed_index()
    credit_spread, credit_direction = get_credit_spreads()
    sector_momentum = get_sector_momentum()

    # Save macro data
    save_macro_data(sp500_df, vix_df, fed_rate, fed_direction, fg_score, fg_rating, credit_spread, credit_direction)

    # Fetch and save per-stock signals
    for symbol in WATCHLIST:
        print(f"\nProcessing {symbol}...")
        short_data = get_short_interest(symbol)
        analyst_data = get_analyst_ratings(symbol)

        sector = STOCK_SECTOR_MAP.get(symbol)
        sector_vs_sp500 = None
        if sector and sector in sector_momentum:
            sector_vs_sp500 = float(sector_momentum[sector]["vs_sp500"])
        save_stock_signals(symbol, sector_vs_sp500, short_data, analyst_data)
        time.sleep(1)

    print("\nAll macro and stock signals saved!")