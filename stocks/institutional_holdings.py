import os
import time
import requests
import psycopg2
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "User-Agent": "TradingSystem/0.1 (contact@example.com)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Major institutional investors to track
BIG_FUNDS = {
    "0001067983": "Berkshire Hathaway",    # Warren Buffett
    "0001166559": "Citadel",               # Ken Griffin
    "0001037389": "Renaissance Technologies", # Jim Simons
    "0001350694": "Bridgewater Associates", # Ray Dalio
    "0001336528": "Soros Fund Management",  # George Soros
    "0001364742": "BlackRock",             # Larry Fink
    "0001061768": "JPMorgan Chase",
    "0001697748": "Cathie Wood ARK Invest",
}


def get_13f_holdings(cik, fund_name):
    print(f"  Fetching 13F for {fund_name}...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.2)

    try:
        response = SESSION.get(url, timeout=30)
        data = response.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            print(f"    No filings found")
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])

        holdings = []

        for i, form in enumerate(forms):
            if form != "13F-HR":
                continue

            filing_date = filing_dates[i] if i < len(filing_dates) else None
            accession = accession_numbers[i] if i < len(accession_numbers) else None

            if not accession or not filing_date:
                continue

            # Parse the 13F XML
            acc_clean = accession.replace("-", "")
            info_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"
            time.sleep(0.2)

            try:
                idx_resp = SESSION.get(info_url, timeout=30)
                idx_data = idx_resp.json()

                for item in idx_data.get("directory", {}).get("item", []):
                    name = item.get("name", "")
                    if "infotable" in name.lower() or name.endswith(".xml"):
                        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{name}"
                        time.sleep(0.2)

                        xml_resp = SESSION.get(xml_url, timeout=30)
                        parsed = parse_13f_xml(xml_resp.text, fund_name, filing_date)
                        holdings.extend(parsed)
                        break

            except Exception as e:
                continue

            # Only get the most recent filing
            break

        print(f"    Found {len(holdings)} holdings")
        return holdings

    except Exception as e:
        print(f"    Failed: {e}")
        return []


def parse_13f_xml(xml_text, fund_name, filing_date):
    import xml.etree.ElementTree as ET

    holdings = []

    try:
        root = ET.fromstring(xml_text)

        for info in root.iter():
            if 'infoTable' in info.tag:
                name_elem = info.find('.//{*}nameOfIssuer')
                cusip_elem = info.find('.//{*}cusip')
                value_elem = info.find('.//{*}value')
                shares_elem = info.find('.//{*}sshPrnamt')
                put_call_elem = info.find('.//{*}putCall')

                if name_elem is not None and value_elem is not None:
                    holdings.append({
                        "fund_name": fund_name,
                        "filing_date": filing_date,
                        "company_name": name_elem.text if name_elem is not None else None,
                        "cusip": cusip_elem.text if cusip_elem is not None else None,
                        "value_thousands": int(value_elem.text) if value_elem is not None else 0,
                        "shares": int(shares_elem.text) if shares_elem is not None else 0,
                        "put_call": put_call_elem.text if put_call_elem is not None else None,
                    })

    except Exception:
        pass

    return holdings


def match_holdings_to_watchlist(holdings):
    from polygon_scraper import WATCHLIST

    # Map common company names to tickers
    name_to_ticker = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA",
        "AMAZON": "AMZN", "TESLA": "TSLA", "META": "META",
        "ALPHABET": "GOOGL", "AMD": "AMD", "ADVANCED MICRO": "AMD",
        "JPMORGAN": "JPM", "GOLDMAN": "GS", "VISA": "V",
        "EXXON": "XOM", "OCCIDENTAL": "OXY", "DEVON": "DVN",
        "SCHLUMBERGER": "SLB", "PFIZER": "PFE", "UNITEDHEALTH": "UNH",
        "WALMART": "WMT", "NIKE": "NKE", "FREEPORT": "FCX",
        "NEWMONT": "NEM", "LOCKHEED": "LMT", "INTEL": "INTC",
        "UBER": "UBER", "PALANTIR": "PLTR", "SOFI": "SOFI",
        "COINBASE": "COIN", "MARATHON DIG": "MARA", "RIOT": "RIOT",
        "SUPER MICRO": "SMCI", "ENPHASE": "ENPH", "LUCID": "LCID",
        "GAMESTOP": "GME", "AMC": "AMC", "CARVANA": "CVNA",
        "UPSTART": "UPST", "AFFIRM": "AFRM", "MICROSTRATEGY": "MSTR",
        "ROBINHOOD": "HOOD", "PROCTER": "PG", "COCA": "KO",
        "JOHNSON": "JNJ", "MCDONALD": "MCD", "AT&T": "T",
        "TAIWAN SEMI": "TSM", "APPLIED OPTO": "AAOI",
        "TRADE DESK": "TTD", "DELTA AIR": "DAL",
        "SALESFORCE": "CRM", "BOSTON SCI": "BSX",
    }

    matched = {}

    for h in holdings:
        company = (h.get("company_name") or "").upper()
        matched_ticker = None

        for name_part, ticker in name_to_ticker.items():
            if name_part in company:
                matched_ticker = ticker
                break

        if matched_ticker and matched_ticker in WATCHLIST:
            if matched_ticker not in matched:
                matched[matched_ticker] = {
                    "symbol": matched_ticker,
                    "total_funds_holding": 0,
                    "total_value_thousands": 0,
                    "total_shares": 0,
                    "fund_names": [],
                }

            matched[matched_ticker]["total_funds_holding"] += 1
            matched[matched_ticker]["total_value_thousands"] += h["value_thousands"]
            matched[matched_ticker]["total_shares"] += h["shares"]
            matched[matched_ticker]["fund_names"].append(h["fund_name"])

    return matched


def save_institutional(matched_data):
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    today = date.today().strftime("%Y-%m-%d")
    total_saved = 0

    for symbol, data in matched_data.items():
        try:
            cursor.execute(
                """INSERT INTO institutional_holdings
                    (symbol, date, funds_holding, total_value_thousands,
                     total_shares, fund_names)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING""",
                (
                    symbol,
                    today,
                    data["total_funds_holding"],
                    data["total_value_thousands"],
                    data["total_shares"],
                    ",".join(data["fund_names"]),
                )
            )
            total_saved += 1
        except Exception:
            conn.rollback()
            continue

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} institutional holding records.")


if __name__ == "__main__":
    all_holdings = []

    for cik, fund_name in BIG_FUNDS.items():
        holdings = get_13f_holdings(cik, fund_name)
        all_holdings.extend(holdings)
        time.sleep(1)

    print(f"\nTotal holdings found: {len(all_holdings)}")

    matched = match_holdings_to_watchlist(all_holdings)

    print(f"\nMatched to watchlist:")
    for symbol, data in sorted(matched.items()):
        print(f"  {symbol}: {data['total_funds_holding']} funds, ${data['total_value_thousands']}K value")
        print(f"    Funds: {', '.join(data['fund_names'])}")

    save_institutional(matched)