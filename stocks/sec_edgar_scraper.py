import os
import time
import requests
import psycopg2
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "User-Agent": "TradingSystem/0.1 (nicksanchez232@gmail.com)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def get_cik(symbol):
    url = "https://www.sec.gov/files/company_tickers.json"
    time.sleep(0.1)
    response = SESSION.get(url, timeout=30)
    data = response.json()

    for key, info in data.items():
        if info.get("ticker", "").upper() == symbol.upper():
            cik = str(info["cik_str"]).zfill(10)
            print(f"Found CIK for {symbol}: {cik}")
            return cik

    print(f"Could not find CIK for {symbol}")
    return None

def get_insider_trades(symbol, days_back=365):
    cik = get_cik(symbol)
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.1)
    response = SESSION.get(url, timeout=30)
    data = response.json()

    trades = []
    recent = data.get("filings", {}).get("recent", {})

    if not recent:
        print(f"No filings found for {symbol}")
        return []

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    cutoff = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    for i, form_type in enumerate(forms):
        if form_type != "4":
            continue

        filing_date = filing_dates[i] if i < len(filing_dates) else None
        if not filing_date or filing_date < cutoff:
            continue

        accession = accession_numbers[i] if i < len(accession_numbers) else None
        primary_doc = primary_docs[i] if i < len(primary_docs) else None

        if accession and primary_doc:
            trade_details = parse_form4(cik, accession, primary_doc, symbol, filing_date)
            trades.extend(trade_details)

    print(f"Found {len(trades)} insider trades for {symbol}")
    return trades

def parse_form4(cik, accession, doc, symbol, filing_date):
    acc_clean = accession.replace("-", "")

    # Try the raw XML URL first (not the HTML rendered version)
    # Remove any xsl prefix path from the doc name
    doc_name = doc.split("/")[-1] if "/" in doc else doc
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc_name}"
    time.sleep(0.1)

    try:
        response = SESSION.get(url, timeout=30)
    except Exception:
        return []

    # If we got HTML instead of XML, try finding the raw XML file
    if "<!DOCTYPE" in response.text[:100] or "<html" in response.text[:100].lower():
        # Look for the XML index to find the raw file
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"
        time.sleep(0.1)
        try:
            idx_resp = SESSION.get(index_url, timeout=30)
            idx_data = idx_resp.json()
            for item in idx_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml") and "primary_doc" not in name.lower():
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{name}"
                    time.sleep(0.1)
                    response = SESSION.get(url, timeout=30)
                    if "<?xml" in response.text[:100]:
                        break
        except Exception:
            return []

    trades = []

    try:
        root = ET.fromstring(response.text)

        insider_name = ""
        insider_title = ""

        name_elem = root.find(".//{http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=}rptOwnerName")
        if name_elem is None:
            name_elem = root.find(".//rptOwnerName")
        if name_elem is not None:
            insider_name = name_elem.text

        title_elem = root.find(".//{http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=}officerTitle")
        if title_elem is None:
            title_elem = root.find(".//officerTitle")
        if title_elem is not None:
            insider_title = title_elem.text or ""

        for txn in root.iter("nonDerivativeTransaction"):
            trade = extract_transaction(txn, symbol, insider_name, insider_title, filing_date)
            if trade:
                trades.append(trade)

    except ET.ParseError:
        pass

    return trades

def extract_transaction(txn, symbol, insider_name, insider_title, filing_date):
    try:
        date_elem = txn.find(".//transactionDate/value")
        trade_date = date_elem.text if date_elem is not None else None

        code_elem = txn.find(".//transactionAcquiredDisposedCode/value")
        acq_disp = code_elem.text if code_elem is not None else None

        shares_elem = txn.find(".//transactionAmounts/transactionShares/value")
        shares = float(shares_elem.text) if shares_elem is not None else 0

        price_elem = txn.find(".//transactionAmounts/transactionPricePerShare/value")
        price = float(price_elem.text) if price_elem is not None and price_elem.text else 0

        owned_elem = txn.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        owned_after = float(owned_elem.text) if owned_elem is not None else 0

        trade_type = "buy" if acq_disp == "A" else "sell"

        return {
            "symbol": symbol,
            "filing_date": filing_date,
            "trade_date": trade_date,
            "insider_name": insider_name,
            "insider_title": insider_title,
            "trade_type": trade_type,
            "shares": int(shares),
            "price_per_share": price,
            "total_value": round(shares * price, 2),
            "shares_owned_after": int(owned_after),
        }

    except Exception:
        return None
    
def save_insider_trades(all_trades):
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    total_saved = 0

    for trade in all_trades:
        cursor.execute(
            "INSERT INTO stocks (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING",
            (trade["symbol"],)
        )

        cursor.execute(
            """INSERT INTO insider_trades
                (symbol, filing_date, trade_date, insider_name, insider_title,
                 trade_type, shares, price_per_share, total_value, shares_owned_after)
            SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM insider_trades
                WHERE symbol = %s
                AND trade_date = %s
                AND insider_name = %s
                AND trade_type = %s
                AND shares = %s
            )""",
            (
                trade["symbol"],
                trade["filing_date"],
                trade["trade_date"],
                trade["insider_name"],
                trade["insider_title"],
                trade["trade_type"],
                trade["shares"],
                trade["price_per_share"],
                trade["total_value"],
                trade["shares_owned_after"],
                trade["symbol"],
                trade["trade_date"],
                trade["insider_name"],
                trade["trade_type"],
                trade["shares"],
            )
        )
        total_saved += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {total_saved} insider trades to database.")
def fetch_all_insider_trades(days_back=365):
    from polygon_scraper import WATCHLIST

    all_trades = []
    total = len(WATCHLIST)

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"[{i}/{total}] Fetching insider trades for {symbol}...")
        trades = get_insider_trades(symbol, days_back)
        all_trades.extend(trades)

    print(f"\nDone! Found {len(all_trades)} total insider trades across {total} stocks.")
    return all_trades

if __name__ == "__main__":
    trades = fetch_all_insider_trades(days_back=365)
    save_insider_trades(trades)