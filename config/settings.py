"""
config/settings.py
Centralised settings loaded from environment / .env file.
Uses pydantic-settings so every value is typed and validated at startup.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_user: str = Field("fraud_user", env="POSTGRES_USER")
    postgres_password: str = Field("fraud_pass", env="POSTGRES_PASSWORD")
    postgres_db: str = Field("fraud_db", env="POSTGRES_DB")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")

    # ── Kafka ──────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_transactions: str = Field("transactions", env="KAFKA_TOPIC_TRANSACTIONS")
    kafka_topic_flagged: str = Field("flagged_transactions", env="KAFKA_TOPIC_FLAGGED")
    kafka_topic_enriched: str = Field("enriched_transactions", env="KAFKA_TOPIC_ENRICHED")
    kafka_topic_reports: str = Field("case_reports", env="KAFKA_TOPIC_REPORTS")
    kafka_consumer_group: str = Field("fraud_detection_group", env="KAFKA_CONSUMER_GROUP")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("fraud_detection", env="MLFLOW_EXPERIMENT_NAME")

    # ── SLA / Thresholds ───────────────────────────────────────────────────────
    detection_latency_ms: int = Field(200, env="DETECTION_LATENCY_MS")
    enrichment_latency_ms: int = Field(500, env="ENRICHMENT_LATENCY_MS")
    fraud_score_threshold: float = Field(0.5, env="FRAUD_SCORE_THRESHOLD")
    high_risk_threshold: float = Field(0.8, env="HIGH_RISK_THRESHOLD")
    retraining_feedback_threshold: int = Field(100, env="RETRAINING_FEEDBACK_THRESHOLD")

    # ── App ────────────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")
    environment: str = Field("development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton — import this everywhere
settings = Settings()
