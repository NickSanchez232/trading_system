import os
import time
import requests
import psycopg2
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()


def get_fundamentals(symbol):
    print(f"Fetching fundamentals for {symbol}...")
    stock = yf.Ticker(symbol)

    try:
        info = stock.info
    except Exception as e:
        print(f"  Failed to get info for {symbol}: {e}")
        return None

    try:
        earnings_hist = stock.earnings_dates
        earnings_beat = None
        earnings_surprise_pct = None
        last_earnings_date = None
        next_earnings_date = None

        if earnings_hist is not None and len(earnings_hist) > 0:
            for idx, row in earnings_hist.iterrows():
                if hasattr(idx, 'date'):
                    earnings_date = idx.date()
                elif hasattr(idx, 'to_pydatetime'):
                    earnings_date = idx.to_pydatetime().date()
                else:
                    continue

                if earnings_date <= date.today():
                    if last_earnings_date is None:
                        last_earnings_date = earnings_date
                        surprise = row.get("Surprise(%)")
                        if pd.notna(surprise):
                            earnings_beat = bool(surprise > 0)
                            earnings_surprise_pct = float(surprise)
                else:
                    if next_earnings_date is None:
                        next_earnings_date = earnings_date
    except Exception as e:
        print(f"  Earnings data error: {e}")
        earnings_beat = None
        earnings_surprise_pct = None
        last_earnings_date = None
        next_earnings_date = None

    days_to_earnings = None
    if next_earnings_date:
        days_to_earnings = (next_earnings_date - date.today()).days

    fcf = info.get("freeCashflow")
    market_cap = info.get("marketCap")
    fcf_yield = None
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap

    fundamentals = {
        "symbol": symbol,
        "date": date.today().strftime("%Y-%m-%d"),
        "revenue": info.get("totalRevenue"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_per_share": info.get("trailingEps"),
        "eps_growth": info.get("earningsGrowth"),
        "profit_margin": info.get("profitMargins"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "sector_avg_pe": None,
        "pe_vs_sector": None,
        "free_cash_flow": fcf,
        "fcf_yield": fcf_yield,
        "debt_to_equity": info.get("debtToEquity"),
        "market_cap": market_cap,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "earnings_beat": earnings_beat,
        "earnings_surprise_pct": earnings_surprise_pct,
        "days_to_next_earnings": days_to_earnings,
    }

    print(f"  Revenue: {fundamentals['revenue']}")
    print(f"  EPS: {fundamentals['earnings_per_share']}")
    print(f"  P/E: {fundamentals['pe_ratio']}")
    print(f"  FCF Yield: {fundamentals['fcf_yield']}")
    print(f"  Earnings Beat: {fundamentals['earnings_beat']}")
    print(f"  Earnings Surprise: {fundamentals['earnings_surprise_pct']}")
    print(f"  Days to Earnings: {fundamentals['days_to_next_earnings']}")

    return fundamentals


def calculate_sector_avg_pe(all_fundamentals):
    sector_pes = {}

    for f in all_fundamentals:
        if f is None:
            continue
        sector = f.get("sector")
        pe = f.get("pe_ratio")
        if sector and pe and pe > 0:
            if sector not in sector_pes:
                sector_pes[sector] = []
            sector_pes[sector].append(pe)

    sector_averages = {}
    for sector, pes in sector_pes.items():
        sector_averages[sector] = sum(pes) / len(pes)
        print(f"  {sector} avg P/E: {sector_averages[sector]:.2f} ({len(pes)} stocks)")

    for f in all_fundamentals:
        if f is None:
            continue
        sector = f.get("sector")
        if sector and sector in sector_averages:
            f["sector_avg_pe"] = sector_averages[sector]
            if f.get("pe_ratio") and f["pe_ratio"] > 0:
                f["pe_vs_sector"] = f["pe_ratio"] / sector_averages[sector]
            else:
                f["pe_vs_sector"] = None
        else:
            f["sector_avg_pe"] = None
            f["pe_vs_sector"] = None

    return all_fundamentals


def save_fundamentals(all_fundamentals):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for f in all_fundamentals:
        if f is None:
            continue

        cursor.execute(
            "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
            (f["symbol"],)
        )

        cursor.execute(
            """INSERT INTO fundamentals
                (symbol, date, revenue, revenue_growth, earnings_per_share,
                 eps_growth, profit_margin, pe_ratio, sector_avg_pe, pe_vs_sector,
                 free_cash_flow, fcf_yield, debt_to_equity,
                 earnings_beat, earnings_surprise_pct, days_to_next_earnings)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING""",
            (
                f["symbol"],
                f["date"],
                f["revenue"],
                f["revenue_growth"],
                f["earnings_per_share"],
                f["eps_growth"],
                f["profit_margin"],
                f["pe_ratio"],
                f["sector_avg_pe"],
                f["pe_vs_sector"],
                f["free_cash_flow"],
                f["fcf_yield"],
                f["debt_to_equity"],
                f["earnings_beat"],
                f["earnings_surprise_pct"],
                f["days_to_next_earnings"],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} fundamental records to database.")


if __name__ == "__main__":
    from polygon_scraper import WATCHLIST

    all_fundamentals = []
    for symbol in WATCHLIST:
        f = get_fundamentals(symbol)
        all_fundamentals.append(f)
        time.sleep(1)

    all_fundamentals = calculate_sector_avg_pe(all_fundamentals)
    save_fundamentals(all_fundamentals)