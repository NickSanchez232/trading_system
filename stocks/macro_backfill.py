import os
import time
import requests
import psycopg2
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()


def backfill_vix():
    print("Backfilling VIX history...")
    vix = yf.Ticker("^VIX")
    hist = vix.history(start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))

    if hist.empty:
        print("  No VIX data returned")
        return None

    df = hist[['Close']].reset_index()
    df.columns = ['date', 'vix_level']
    df['date'] = pd.to_datetime(df['date']).dt.date

    df['vix_ma_5'] = df['vix_level'].rolling(window=5).mean()
    df['vix_direction'] = df['vix_level'] - df['vix_ma_5']

    print(f"  Got {len(df)} days of VIX data")
    return df

def backfill_fed_rate():
    print("Backfilling Fed rate history...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "FEDFUNDS",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "observation_start": "2015-01-01",
        "sort_order": "asc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            print("  No Fed rate data returned")
            return None

        records = []
        prev_rate = None
        for obs in observations:
            try:
                rate = float(obs["value"])
                direction = "stable"
                if prev_rate is not None:
                    if rate > prev_rate:
                        direction = "rising"
                    elif rate < prev_rate:
                        direction = "falling"
                records.append({
                    "date": obs["date"],
                    "fed_rate": rate,
                    "fed_rate_direction": direction,
                })
                prev_rate = rate
            except ValueError:
                continue

        print(f"  Got {len(records)} months of Fed rate data")
        return records

    except Exception as e:
        print(f"  Fed rate backfill failed: {e}")
        return None
    
def backfill_credit_spreads():
    print("Backfilling credit spread history...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "BAMLH0A0HYM2",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "observation_start": "2015-01-01",
        "sort_order": "asc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            print("  No credit spread data returned")
            return None

        records = []
        prev_spread = None
        for obs in observations:
            try:
                spread = float(obs["value"])
                direction = "stable"
                if prev_spread is not None:
                    if spread > prev_spread:
                        direction = "widening"
                    elif spread < prev_spread:
                        direction = "tightening"
                records.append({
                    "date": obs["date"],
                    "credit_spread": spread,
                    "credit_spread_direction": direction,
                })
                prev_spread = spread
            except ValueError:
                continue

        print(f"  Got {len(records)} days of credit spread data")
        return records

    except Exception as e:
        print(f"  Credit spread backfill failed: {e}")
        return None
    
def backfill_sp500():
    print("Backfilling S&P 500 history...")
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))

    if hist.empty:
        print("  No S&P 500 data returned")
        return None

    df = hist[['Close']].reset_index()
    df.columns = ['date', 'sp500_level']
    df['date'] = pd.to_datetime(df['date']).dt.date

    df['ma_50'] = df['sp500_level'].rolling(window=50).mean()
    df['ma_200'] = df['sp500_level'].rolling(window=200).mean()

    def classify_regime(row):
        if pd.isna(row['ma_50']) or pd.isna(row['ma_200']):
            return None
        if row['sp500_level'] > row['ma_50'] > row['ma_200']:
            return 'bull'
        elif row['sp500_level'] < row['ma_50'] < row['ma_200']:
            return 'bear'
        else:
            return 'sideways'

    df['market_regime'] = df.apply(classify_regime, axis=1)

    print(f"  Got {len(df)} days of S&P 500 data")
    return df

def backfill_treasury_yield():
    print("Backfilling 10-Year Treasury Yield...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DGS10",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "observation_start": "2015-01-01",
        "sort_order": "asc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        records = {}
        for obs in observations:
            try:
                records[obs["date"]] = float(obs["value"])
            except ValueError:
                continue

        print(f"  Got {len(records)} days of Treasury yield data")
        return records

    except Exception as e:
        print(f"  Treasury yield backfill failed: {e}")
        return {}


def backfill_unemployment():
    print("Backfilling Unemployment Rate...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "UNRATE",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "observation_start": "2015-01-01",
        "sort_order": "asc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        records = {}
        for obs in observations:
            try:
                records[obs["date"]] = float(obs["value"])
            except ValueError:
                continue

        print(f"  Got {len(records)} months of unemployment data")
        return records

    except Exception as e:
        print(f"  Unemployment backfill failed: {e}")
        return {}


def backfill_cpi():
    print("Backfilling CPI Inflation...")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": os.getenv("FRED_API_KEY", ""),
        "file_type": "json",
        "observation_start": "2015-01-01",
        "sort_order": "asc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        observations = data.get("observations", [])

        records = {}
        prev_cpi = None
        for obs in observations:
            try:
                cpi = float(obs["value"])
                yoy_change = None
                if prev_cpi and prev_cpi > 0:
                    yoy_change = (cpi - prev_cpi) / prev_cpi
                records[obs["date"]] = {
                    "cpi": cpi,
                    "cpi_yoy_change": yoy_change,
                }
                prev_cpi = cpi
            except ValueError:
                continue

        print(f"  Got {len(records)} months of CPI data")
        return records

    except Exception as e:
        print(f"  CPI backfill failed: {e}")
        return {}


def backfill_sector_etfs():
    print("Backfilling Sector ETF history...")
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

    all_sector_data = {}

    for etf, name in sectors.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))
            if hist.empty:
                continue

            df = hist[['Close']].reset_index()
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['momentum_30d'] = df['close'].pct_change(periods=21)

            for _, row in df.iterrows():
                d = str(row['date'])
                if d not in all_sector_data:
                    all_sector_data[d] = {}
                all_sector_data[d][name] = float(row['momentum_30d']) if pd.notna(row['momentum_30d']) else None

            print(f"  {name}: {len(df)} days")
            time.sleep(1)
        except Exception as e:
            print(f"  {name} failed: {e}")

    print(f"  Total dates with sector data: {len(all_sector_data)}")
    return all_sector_data


def save_macro_backfill(vix_df, sp500_df, fed_records, credit_records, treasury_data, unemployment_data, cpi_data, sector_data):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    if vix_df is not None and sp500_df is not None:
        merged = vix_df.merge(sp500_df[['date', 'sp500_level', 'market_regime']], on='date', how='outer')

        fed_dict = {}
        if fed_records:
            for r in fed_records:
                fed_dict[r["date"]] = r

        credit_dict = {}
        if credit_records:
            for r in credit_records:
                credit_dict[r["date"]] = r

        for _, row in merged.iterrows():
            d = str(row['date'])
            fed = fed_dict.get(d, {})
            credit = credit_dict.get(d, {})
            treasury = treasury_data.get(d)
            unemployment = unemployment_data.get(d)
            cpi = cpi_data.get(d, {})
            sectors = sector_data.get(d, {})

            try:
                cursor.execute(
                    """INSERT INTO macro_data
                        (date, sp500_level, market_regime, vix_level, vix_direction,
                         fed_rate, fed_rate_direction, credit_spread, credit_spread_direction,
                         treasury_yield_10y, unemployment_rate, cpi_level, cpi_yoy_change,
                         sector_tech_momentum, sector_financial_momentum, sector_energy_momentum,
                         sector_healthcare_momentum, sector_consumer_cyc_momentum,
                         sector_consumer_def_momentum, sector_industrial_momentum,
                         sector_materials_momentum)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        sp500_level = COALESCE(EXCLUDED.sp500_level, macro_data.sp500_level),
                        market_regime = COALESCE(EXCLUDED.market_regime, macro_data.market_regime),
                        vix_level = COALESCE(EXCLUDED.vix_level, macro_data.vix_level),
                        vix_direction = COALESCE(EXCLUDED.vix_direction, macro_data.vix_direction),
                        fed_rate = COALESCE(EXCLUDED.fed_rate, macro_data.fed_rate),
                        fed_rate_direction = COALESCE(EXCLUDED.fed_rate_direction, macro_data.fed_rate_direction),
                        credit_spread = COALESCE(EXCLUDED.credit_spread, macro_data.credit_spread),
                        credit_spread_direction = COALESCE(EXCLUDED.credit_spread_direction, macro_data.credit_spread_direction),
                        treasury_yield_10y = COALESCE(EXCLUDED.treasury_yield_10y, macro_data.treasury_yield_10y),
                        unemployment_rate = COALESCE(EXCLUDED.unemployment_rate, macro_data.unemployment_rate),
                        cpi_level = COALESCE(EXCLUDED.cpi_level, macro_data.cpi_level),
                        cpi_yoy_change = COALESCE(EXCLUDED.cpi_yoy_change, macro_data.cpi_yoy_change),
                        sector_tech_momentum = COALESCE(EXCLUDED.sector_tech_momentum, macro_data.sector_tech_momentum),
                        sector_financial_momentum = COALESCE(EXCLUDED.sector_financial_momentum, macro_data.sector_financial_momentum),
                        sector_energy_momentum = COALESCE(EXCLUDED.sector_energy_momentum, macro_data.sector_energy_momentum),
                        sector_healthcare_momentum = COALESCE(EXCLUDED.sector_healthcare_momentum, macro_data.sector_healthcare_momentum),
                        sector_consumer_cyc_momentum = COALESCE(EXCLUDED.sector_consumer_cyc_momentum, macro_data.sector_consumer_cyc_momentum),
                        sector_consumer_def_momentum = COALESCE(EXCLUDED.sector_consumer_def_momentum, macro_data.sector_consumer_def_momentum),
                        sector_industrial_momentum = COALESCE(EXCLUDED.sector_industrial_momentum, macro_data.sector_industrial_momentum),
                        sector_materials_momentum = COALESCE(EXCLUDED.sector_materials_momentum, macro_data.sector_materials_momentum)
                    """,
                    (
                        row['date'],
                        float(row['sp500_level']) if pd.notna(row.get('sp500_level')) else None,
                        row.get('market_regime'),
                        float(row['vix_level']) if pd.notna(row.get('vix_level')) else None,
                        float(row['vix_direction']) if pd.notna(row.get('vix_direction')) else None,
                        fed.get('fed_rate'),
                        fed.get('fed_rate_direction'),
                        credit.get('credit_spread'),
                        credit.get('credit_spread_direction'),
                        treasury,
                        unemployment,
                        cpi.get('cpi') if isinstance(cpi, dict) else None,
                        cpi.get('cpi_yoy_change') if isinstance(cpi, dict) else None,
                        sectors.get('Technology'),
                        sectors.get('Financial'),
                        sectors.get('Energy'),
                        sectors.get('Healthcare'),
                        sectors.get('Consumer Cyclical'),
                        sectors.get('Consumer Defensive'),
                        sectors.get('Industrials'),
                        sectors.get('Materials'),
                    )
                )
                total_saved += 1
            except Exception as e:
                conn.rollback()
                continue

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} macro records to database.")


if __name__ == "__main__":
    vix_df = backfill_vix()
    sp500_df = backfill_sp500()
    fed_records = backfill_fed_rate()
    credit_records = backfill_credit_spreads()
    treasury_data = backfill_treasury_yield()
    unemployment_data = backfill_unemployment()
    cpi_data = backfill_cpi()
    sector_data = backfill_sector_etfs()
    save_macro_backfill(vix_df, sp500_df, fed_records, credit_records, treasury_data, unemployment_data, cpi_data, sector_data)