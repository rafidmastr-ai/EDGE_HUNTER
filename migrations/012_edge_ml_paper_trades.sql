-- Experimental EDGE ML paper trades (research signals only; never part of the main signal).
CREATE TABLE IF NOT EXISTS edge_ml_paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key TEXT NOT NULL,
    variant TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_time INTEGER NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry_time INTEGER NOT NULL,
    entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    exit_time INTEGER,
    exit REAL,
    exit_reason TEXT,
    r_gross REAL,
    r_net REAL,
    swap_nights INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_key, signal_time)
);

CREATE INDEX IF NOT EXISTS idx_edge_ml_paper_trades_symbol_time ON edge_ml_paper_trades(symbol, signal_time);
