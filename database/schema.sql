CREATE TABLE IF NOT EXISTS stocks (
    symbol          VARCHAR(10) PRIMARY KEY,
    company_name    VARCHAR(255),
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    market_cap      BIGINT,
    added_at        TIMESTAMP DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS daily_prices (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL REFERENCES stocks(symbol),
    date            DATE NOT NULL,
    open            NUMERIC(12, 4),
    high            NUMERIC(12, 4),
    low             NUMERIC(12, 4),
    close           NUMERIC(12, 4),
    volume          BIGINT,
    vwap            NUMERIC(12, 4),
    num_trades      INTEGER,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date ON daily_prices(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date DESC);