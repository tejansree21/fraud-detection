# Multi-Agent Fraud Detection Pipeline
**MSc Artificial Intelligence — NCI Dublin, 2026**
Tejan Sree Challa

A real-time fraud detection system using three specialised AI agents:
Anomaly Detector → Context Enricher → Case Reporter (Groq/Llama 3.3 70B)
with a continuous feedback retraining loop.

---

## Project Structure

```
fraud-detection/
├── agents/
│   ├── base_agent.py         ← Abstract base all agents inherit
│   ├── anomaly_detector.py   ← Stage 1: ML fraud scoring (stub → Phase 2)
│   ├── context_enricher.py   ← Stage 2: Feature enrichment (stub → Phase 3)
│   └── case_reporter.py      ← Stage 3: Groq LLM report generation
├── orchestrator/
│   └── orchestrator.py       ← Pipeline coordinator + feedback loop
├── pipeline/
│   └── schemas.py            ← Shared Pydantic data models
├── config/
│   └── settings.py           ← Typed settings from .env
├── scripts/
│   └── init_db.sql           ← PostgreSQL schema
├── tests/
│   └── test_pipeline.py      ← End-to-end pipeline tests
├── main.py                   ← FastAPI REST API
├── docker-compose.yml        ← Kafka, PostgreSQL, Redis, MLflow
├── requirements.txt
└── .env.example
```

---

## Quick Start (Windows + Docker)

### 1. Clone & configure

```bash
git clone <your-repo>
cd fraud-detection

# Copy env file and fill in your Groq API key
copy .env.example .env
```

Edit `.env` and set your `GROQ_API_KEY`.
Get a free key at: https://console.groq.com

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts:
| Service | URL |
|---|---|
| Kafka | localhost:9092 |
| Kafka UI | http://localhost:8080 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |
| MLflow | http://localhost:5000 |

Wait ~30 seconds for all services to be healthy.

### 3. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 5. Run tests

```bash
pytest tests/ -v
```

---

## API Endpoints

### Process a transaction
```bash
curl -X POST http://localhost:8000/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_42",
    "amount": 1500.00,
    "merchant": "CryptoExchange",
    "merchant_category": "unknown",
    "location": "unknown",
    "device_id": "dev_new_999"
  }'
```

### Get a case report
```bash
curl http://localhost:8000/report/{transaction_id}
```

### Submit investigator feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "...",
    "investigator_id": "analyst_1",
    "decision": "CONFIRMED_FRAUD"
  }'
```

### Health check
```bash
curl http://localhost:8000/health
```

---

## Build Phases

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Done | Scaffold, Docker, base agent, orchestrator |
| 1 | 🔜 Next | Kafka producer + PaySim/CC Fraud data streaming |
| 2 | 🔜 | Anomaly Detector: PyTorch autoencoder + Isolation Forest |
| 3 | 🔜 | Context Enricher: real Redis feature store |
| 4 | 🔜 | Case Reporter: vector similarity for similar past cases |
| 5 | 🔜 | PostgreSQL persistence at each pipeline stage |
| 6 | 🔜 | Feedback loop + ADWIN drift detection + MLflow retraining |
| 7 | 🔜 | Streamlit investigator dashboard |

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API — Llama 3.3 70B (free tier) |
| Stream | Apache Kafka (Docker) |
| ML | PyTorch, Scikit-learn |
| Orchestration | Custom + LangChain (Phase 4+) |
| API | FastAPI |
| Storage | PostgreSQL + Redis |
| Tracking | MLflow |
| UI | Streamlit (Phase 7) |
| Infra | Docker Compose |
