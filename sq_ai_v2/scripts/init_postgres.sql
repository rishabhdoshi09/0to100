-- Initialisation SQL run by docker-compose on first PostgreSQL startup

CREATE TABLE IF NOT EXISTS earnings (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20)  NOT NULL,
    report_date     DATE         NOT NULL,
    eps_actual      FLOAT,
    eps_est         FLOAT,
    revenue_actual  FLOAT,
    revenue_est     FLOAT,
    surprise_pct    FLOAT,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS macro_indicators (
    id              SERIAL PRIMARY KEY,
    indicator       VARCHAR(50)  NOT NULL,
    release_date    DATE         NOT NULL,
    actual          FLOAT,
    expected        FLOAT,
    surprise        FLOAT,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(50)  NOT NULL,
    train_start     DATE,
    train_end       DATE,
    val_sharpe      FLOAT,
    val_accuracy    FLOAT,
    artifact_path   TEXT,
    is_active       BOOLEAN      DEFAULT FALSE,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_log (
    id              SERIAL PRIMARY KEY,
    log_date        DATE         NOT NULL,
    symbol          VARCHAR(20),
    pnl             FLOAT,
    win_rate        FLOAT,
    sharpe          FLOAT,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_earnings_symbol ON earnings(symbol);
CREATE INDEX IF NOT EXISTS ix_earnings_date   ON earnings(report_date);
CREATE INDEX IF NOT EXISTS ix_macro_indicator ON macro_indicators(indicator);
CREATE INDEX IF NOT EXISTS ix_model_active    ON model_registry(is_active);
CREATE INDEX IF NOT EXISTS ix_perf_date       ON performance_log(log_date);
