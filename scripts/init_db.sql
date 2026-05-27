-- ── Transactions table ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    transaction_id  VARCHAR(64) UNIQUE NOT NULL,
    user_id         VARCHAR(64) NOT NULL,
    amount          DECIMAL(12, 2) NOT NULL,
    merchant        VARCHAR(128),
    merchant_category VARCHAR(64),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    location        VARCHAR(128),
    device_id       VARCHAR(64),
    ip_address      VARCHAR(45),
    raw_payload     JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Flagged transactions (Anomaly Detector output) ─────────────────────────────
CREATE TABLE IF NOT EXISTS flagged_transactions (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64) REFERENCES transactions(transaction_id),
    fraud_score         FLOAT NOT NULL,
    anomaly_vector      JSONB,           -- explanation vector from detector
    detector_version    VARCHAR(32),
    flagged_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Enriched transactions (Context Enricher output) ───────────────────────────
CREATE TABLE IF NOT EXISTS enriched_transactions (
    id                      SERIAL PRIMARY KEY,
    transaction_id          VARCHAR(64) REFERENCES transactions(transaction_id),
    merchant_risk_score     FLOAT,
    geo_consistency_score   FLOAT,
    device_match            BOOLEAN,
    velocity_hour           INT,
    velocity_day            INT,
    user_behaviour_score    FLOAT,
    enriched_payload        JSONB,
    enriched_at             TIMESTAMPTZ DEFAULT NOW()
);

-- ── Case reports (Case Reporter output) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS case_reports (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64) REFERENCES transactions(transaction_id),
    report_text         TEXT NOT NULL,
    recommended_action  VARCHAR(16) CHECK (recommended_action IN ('BLOCK','REVIEW','ALLOW')),
    confidence_score    FLOAT,
    similar_cases       JSONB,
    generated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Investigator feedback (feeds retraining loop) ─────────────────────────────
CREATE TABLE IF NOT EXISTS investigator_feedback (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64) REFERENCES transactions(transaction_id),
    investigator_id     VARCHAR(64),
    decision            VARCHAR(16) CHECK (decision IN ('CONFIRMED_FRAUD','FALSE_POSITIVE','NEEDS_REVIEW')),
    notes               TEXT,
    decided_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Model registry metadata ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(64) NOT NULL,
    version         VARCHAR(32) NOT NULL,
    mlflow_run_id   VARCHAR(64),
    auc_pr          FLOAT,
    f1_score        FLOAT,
    is_active       BOOLEAN DEFAULT FALSE,
    trained_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_flagged_transaction_id ON flagged_transactions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_transaction_id ON investigator_feedback(transaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_decided_at ON investigator_feedback(decided_at);
