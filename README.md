# 🇿🇦 sa-fraud-shield
### Production-Grade SA Fraud Detection — GNN + Streaming + Federated Learning + POPIA-Compliant XAI

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-140%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 🎯 Project Overview

**Business Problem:**  
South African banks lost **R2.7 billion to financial crime in 2024**, with digital banking fraud surging 86% to ~98,000 incidents. Every R1 lost costs SA banks **R4.52** in total when factoring in investigation, recovery, and customer churn. Yet no SA bank has deployed graph neural networks, federated learning, or transformer-based models — they're still running FICO Falcon (circa 2011) and vendor-supplied solutions.

**Solution:**  
A **production-grade fraud detection system** that demonstrates what SA banks actually need but haven't shipped yet:
- **AUC 0.9922** SIM swap detection model — SA's dominant fraud vector (60% of mobile breaches)
- **Graph Neural Network** fraud ring detector on heterogeneous transaction graphs
- **Real-time Kafka streaming** with sub-100ms scoring latency
- **Federated learning** across 5 simulated SA banks with differential privacy (POPIA-compliant)
- **LLM-generated explanations** in English and isiZulu — satisfying POPIA Section 71

**Why This Matters:**  
SA banks plan to invest **R30M+ each in AI in 2026**. This project demonstrates the exact stack they need — and that no Kaggle notebook comes close to replicating.

---

## 📊 Key Results

| Metric | Value | Context |
|--------|-------|---------|
| **SIM swap AUC** | **0.9922** | Temporal cross-validation, 34 features |
| **GNN AUC** | 0.6450 | GraphSAGE on heterogeneous fraud graph |
| **API latency** | <10ms | FastAPI + LightGBM serving |
| **Test coverage** | 140 tests | Unit / integration / e2e — all passing |
| **Federated rounds** | 5 banks | DPFedAvg + ε-differential privacy |
| **Explainability** | Full | SHAP + LLM narratives (EN + isiZulu) |

---

## 🇿🇦 South African Context

### The Fraud Landscape

```
SA Digital Banking Fraud (2024):
├── Total losses: R2.7B (+86% YoY)
├── Banking app fraud: 65% of all cases (~64,000 incidents)
├── Card-not-present: 85.6% of gross credit card losses
├── SIM swap: 60% of mobile banking breaches
│   ├── ~R5.3B/year in telecoms-linked fraud
│   ├── Average loss: R10,000 per incident
│   └── 15% involve telecom/bank employee collusion
└── Cost multiplier: R4.52 lost per R1 of fraud
```

### What Makes SA Fraud Unique

- **SIM swap as primary attack vector** — fraudsters port the victim's number to intercept OTPs, then drain accounts within hours. No international fraud model is built for this.
- **Load shedding noise** — Eskom rolling blackouts create legitimate connectivity gaps that mimic SIM swap indicators. This model cross-references outage schedules to distinguish infrastructure anomalies from genuine fraud.
- **PayShap graph structure** — SARB's instant payment system uses phone-number ShapIDs (+27XXXXXXXXX@bank), forming a natural bipartite graph perfect for GNN-based mule network detection.

### Regulatory Environment

- **POPIA Section 71** — Restricts purely automated decisions with significant effects. Requires banks to provide "sufficient information about the underlying logic." Every decision ships with SHAP values and an LLM-generated narrative in English and isiZulu.
- **FICA** — Suspicious transaction reports, 48-hour cash reporting, 10-year record retention.
- **SA removed from FATF grey list (Oct 2025)** — Enforcement intensified. Sasfin Bank fined R209M in 2024.

---

## 🏗️ Architecture

```
Synthetic SA Data (Faker en_ZA)
        │
        ▼
  Kafka Producer  ──►  Enrichment Consumer  ──►  Velocity Checker (Redis)
                                │
                                ▼
              ┌──────────────────────────────────┐
              │          FastAPI /score           │
              │                                   │
              │   SIM Swap Model (LightGBM)        │  AUC 0.9922
              │   + GNN Fraud Ring Detector        │  AUC 0.6450
              │   → Ensemble (0.6 SIM / 0.4 GNN)  │
              │   → APPROVE / STEP_UP / BLOCK      │
              └──────────────────────────────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
            SHAP Explainer           LLM Narrative
            (34-feature ΔAUC)   (EN + isiZulu, POPIA §71)
                   │
                   ▼
         Prometheus + Grafana
         (PSI drift monitoring)
```

---

## 📁 Project Structure

```
sa-fraud-shield/
├── data_generation/          # SA-locale synthetic data (Faker en_ZA)
│   ├── generators/           # Identity, banking, transactions
│   ├── fraud_patterns/       # SIM swap sequences, fraud ring injection
│   └── graph_builder.py      # Heterogeneous graph for GNN
├── models/
│   ├── sim_swap/             # LightGBM SIM swap detector (AUC 0.9922)
│   ├── gnn/                  # GraphSAGE fraud ring detector
│   └── federated/            # DPFedAvg across 5 SA bank shards
├── streaming/                # Kafka producer + enrichment consumer
├── api/                      # FastAPI — /score, /explain, /drift, /metrics
├── explainability/           # SHAP + GNN attribution + LLM narratives
├── monitoring/               # PSI drift detector + Prometheus metrics
├── infra/
│   ├── terraform/            # AWS VPC, MSK, ECS Fargate, ElastiCache, S3, IAM
│   └── docker/               # Prometheus + Grafana configs
├── notebooks/
│   ├── 01_graph_exploration.ipynb
│   └── 02_sim_swap_model.ipynb
└── tests/                    # 140 tests — unit / integration / e2e
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Kafka + Redis)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/mmachelane/sa-fraud-shield.git
cd sa-fraud-shield

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Generate Data & Train

```bash
# Generate synthetic SA transaction data
python -m data_generation.scripts.generate_training_data

# Train SIM swap model
python -m models.sim_swap.train

# Train GNN fraud ring detector
python -m models.gnn.train
```

### Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Score a Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "timestamp": "2026-03-15T10:30:00Z",
      "sender_account_id": "acc-001",
      "amount_zar": 45000,
      "payment_rail": "PAYSHAP",
      "sim_swap_detected": true,
      "sim_swap_timestamp": "2026-03-15T10:25:00Z"
    }
  }'
```

**Response:**
```json
{
  "transaction_id": "...",
  "ensemble_score": 0.91,
  "sim_swap_score": 0.94,
  "decision": "BLOCK",
  "alert": {
    "severity": "CRITICAL",
    "triggered_rules": ["ensemble_score=0.910 >= 0.7"]
  },
  "latency_ms": 4.2
}
```

### Run Tests

```bash
pytest tests/
# 140 passed in 44s
```

---

## 📈 Build Phases

| Phase | Component | Highlight |
|-------|-----------|-----------|
| 1–2 | Scaffold + shared library | Pydantic schemas, SA validators, constants |
| 3 | Synthetic data generation | SA-locale Faker, PayShap IDs, load shedding injection |
| 4 | SIM swap model | LightGBM, 34 features, temporal CV — **AUC 0.9922** |
| 5 | GNN fraud ring detector | GraphSAGE on heterogeneous graph — AUC 0.6450 |
| 6 | Federated learning | DPFedAvg, ε-differential privacy, 5 SA bank shards |
| 7 | Streaming pipeline | Kafka → enrichment consumer → Redis velocity checker |
| 8 | FastAPI scoring layer | `/score`, `/explain`, `/debug-features` |
| 9 | Explainability | SHAP + GNN attribution + LLM narratives (EN + isiZulu) |
| 10 | Monitoring | PSI drift detector, Prometheus metrics, Grafana dashboard |
| 11 | AWS Terraform | VPC, MSK, ECS Fargate, ElastiCache, S3, IAM |
| 12 | Test suite | **140 tests** — unit / integration / e2e, all passing |
| 13 | SIM swap notebook | Feature distributions, temporal CV, AUC/PR curves, SHAP waterfall |

---

## 🔍 Key Capabilities

### 1. SIM Swap Detection (AUC 0.9922)
SA's signature fraud type — 60% of mobile banking breaches. 34 engineered features capture:
- Swap-to-transaction timing (minutes since SIM swap)
- Device fingerprint changes
- Velocity spikes post-swap
- OTP request patterns
- Load shedding coincidence (SA-specific noise filter)

### 2. GNN Fraud Ring Detector
GraphSAGE on a heterogeneous graph where node types are accounts, devices, and merchants — edge types are transactions, shared devices, and shared addresses. Fraudsters camouflage individual transaction features but leave relational traces that only graphs capture.

### 3. Federated Learning (POPIA-Compliant)
Five simulated SA banks train local LightGBM models with FedAvg aggregation and ε-differential privacy. Banks contribute fraud signal without sharing raw customer data — directly addressing POPIA's cross-institutional data sharing constraints.

### 4. POPIA Section 71 Explainability
Every BLOCK or STEP_UP decision includes:
- **SHAP values** — per-feature attributions (e.g., `time_since_sim_swap_minutes: +0.43`)
- **LLM narrative (English)** — "This R45,000 PayShap transfer was flagged because it occurred 5 minutes after a SIM swap was detected on the registered mobile number..."
- **LLM narrative (isiZulu)** — same explanation in isiZulu for POPIA §71 compliance

### 5. Real-Time Streaming
```
Transaction events → Kafka topic
        ↓
Enrichment consumer (asyncio)
        ↓
Redis velocity check (5min / 1hr / 24hr windows)
        ↓
FastAPI /score → ensemble decision in <10ms
```

---

## ☁️ Infrastructure (AWS)

Terraform IaC targeting SA banks' preferred cloud (Capitec, TymeBank, Standard Bank all run on AWS):

| Component | AWS Service |
|-----------|------------|
| Event streaming | Amazon MSK (Managed Kafka) |
| Model serving | ECS Fargate |
| Feature store | ElastiCache (Redis) |
| Artifact storage | S3 |
| Networking | VPC — public/private subnets, NAT gateway |
| Permissions | IAM roles — ECS task, MSK access, S3 read |

---

## 🧪 Test Suite

```
tests/
├── unit/
│   ├── test_sa_validators.py     # SA ID, phone, account, PayShap, postal code
│   ├── test_schemas.py           # Pydantic schema validation + cross-field rules
│   ├── test_load_shedding.py     # Outage detection + feature extraction
│   └── test_drift_detector.py   # PSI computation + stable/warning/drift logic
├── integration/
│   └── test_api.py               # All endpoints — health, score, drift, metrics
└── e2e/
    └── test_score_flow.py        # Full pipeline — all 5 payment rails, SIM swap path
```

**140 tests, 0 failures.**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML — tabular | LightGBM |
| ML — graph | PyTorch Geometric (GraphSAGE) |
| Federated learning | Flower (flwr) + differential privacy |
| API | FastAPI + Uvicorn |
| Streaming | Kafka (aiokafka) |
| Feature store | Redis |
| Explainability | SHAP + LiteLLM |
| MLOps | MLflow |
| Monitoring | Prometheus + Grafana |
| Infrastructure | Terraform + Docker |
| Testing | pytest + pytest-asyncio |

---

## 🏆 What Sets This Apart

| Capability | sa-fraud-shield |
|-----------|----------------|
| Graph neural networks | ✅ GraphSAGE fraud ring detector |
| Real-time streaming | ✅ Kafka + async enrichment pipeline |
| Federated learning | ✅ DPFedAvg across 5 SA bank shards |
| SIM swap detection | ✅ AUC 0.9922 — SA's #1 fraud type |
| Load shedding awareness | ✅ Eskom schedule integration |
| POPIA Section 71 | ✅ SHAP + LLM narratives (EN + isiZulu) |
| AWS infrastructure (IaC) | ✅ Terraform — full production stack |
| Test suite | ✅ 140 tests — unit / integration / e2e |

---

*Built by [Mmachelane Karabo Moswane](https://github.com/mmachelane)*
