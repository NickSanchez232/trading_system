import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

LOCAL_DB = os.getenv("DATABASE_URL")
RAILWAY_DB = os.getenv("RAILWAY_DATABASE_URL")

TABLES_TO_SYNC = [
    "stocks",
    "daily_prices",
    "insider_trades",
    "insider_signals",
    "technicals",
    "fundamentals",
    "macro_data",
    "stock_signals",
    "sentiment",
    "congress_trades",
    "labels",
    "relative_value",
    "institutional_holdings",
    "buyback_signals",
]


def sync_table(table_name, railway_cursor, local_cursor):
    print(f"Syncing {table_name}...")

    # Get column names from railway
    railway_cursor.execute(f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = '{table_name}' AND column_name != 'id' AND column_name != 'created_at'
        ORDER BY ordinal_position
    """)
    columns = [row[0] for row in railway_cursor.fetchall()]

    if not columns:
        print(f"  No columns found for {table_name}, skipping")
        return 0

    columns_str = ", ".join(columns)

    # Get all data from railway
    railway_cursor.execute(f"SELECT {columns_str} FROM {table_name}")
    rows = railway_cursor.fetchall()

    if not rows:
        print(f"  No data in {table_name}")
        return 0

    # Build insert query with conflict handling
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    inserted = 0
    for row in rows:
        try:
            local_cursor.execute(insert_sql, row)
            inserted += 1
        except Exception as e:
            pass

    print(f"  Synced {inserted} rows for {table_name}")
    return inserted


def run_sync():
    print("Connecting to Railway database...")
    railway_conn = psycopg2.connect(RAILWAY_DB)
    railway_cursor = railway_conn.cursor()

    print("Connecting to local database...")
    local_conn = psycopg2.connect(LOCAL_DB)
    local_cursor = local_conn.cursor()

    total = 0
    for table in TABLES_TO_SYNC:
        count = sync_table(table, railway_cursor, local_cursor)
        total += count

    local_conn.commit()
    local_cursor.close()
    local_conn.close()
    railway_cursor.close()
    railway_conn.close()

    print(f"\nSync complete! {total} total rows synced.")


if __name__ == "__main__":
    run_sync()