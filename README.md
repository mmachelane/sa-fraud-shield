# sa-fraud-shield

> Production-grade SA fraud detection — GNN + streaming + federated learning + POPIA-compliant XAI

South African banks lost **R2.7 billion to financial crime in 2024**, with digital banking fraud surging 86% to ~98,000 incidents. No SA bank has deployed graph neural networks, federated learning, or transformer-based models. This project builds what SA banks actually need but haven't shipped yet.

---

## What makes this different

| Capability | sa-fraud-shield | Typical Kaggle project |
|---|---|---|
| Graph neural networks (GNN) | ✅ GraphSAGE fraud ring detector | ❌ |
| Real-time streaming | ✅ Kafka + async consumer pipeline | ❌ Batch only |
| Federated learning | ✅ DPFedAvg across 5 SA bank shards | ❌ |
| SIM swap detection | ✅ LightGBM — AUC 0.9922 | ❌ |
| Load shedding awareness | ✅ Eskom schedule integration | ❌ |
| POPIA Section 71 compliance | ✅ SHAP + LLM narratives (EN + isiZulu) | ❌ |
| AWS infrastructure (IaC) | ✅ Terraform — VPC, MSK, ECS, ElastiCache | ❌ |
| MLOps pipeline | ✅ MLflow, Docker, Prometheus, Grafana | ❌ Notebook only |
| Test suite | ✅ 140 tests — unit / integration / e2e | ❌ |

---

## Architecture

```
Synthetic Data (SA-locale Faker)
        │
        ▼
  Kafka Producer  ──►  Enrichment Consumer  ──►  Velocity Checker (Redis)
                                │
                                ▼
              ┌─────────────────────────────────┐
              │         FastAPI /score           │
              │                                  │
              │  SIM Swap Model (LightGBM)       │  AUC 0.9922
              │  + GNN Fraud Ring Detector       │  AUC 0.6450
              │  → Ensemble (0.6 / 0.4)          │
              │  → APPROVE / STEP_UP / BLOCK     │
              └─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             SHAP Explainer          LLM Narrative
             (feature ΔAUC)     (EN + isiZulu, POPIA §71)
                    │
                    ▼
          Prometheus + Grafana (PSI drift monitoring)
```

---

## Project structure

```
sa-fraud-shield/
├── data_generation/      # SA-locale synthetic data (Faker en_ZA)
│   ├── generators/       # Identity, banking, transactions
│   ├── fraud_patterns/   # SIM swap sequences, fraud ring injection
│   └── graph_builder.py  # Heterogeneous graph for GNN
├── models/
│   ├── sim_swap/         # LightGBM SIM swap detector (AUC 0.9922)
│   ├── gnn/              # GraphSAGE fraud ring detector
│   └── federated/        # DPFedAvg across 5 SA bank shards
├── streaming/            # Kafka producer + enrichment consumer
├── api/                  # FastAPI — /score, /explain, /drift, /metrics
├── explainability/       # SHAP + GNN attribution + LLM narratives
├── monitoring/           # PSI drift detector + Prometheus metrics
├── infra/
│   ├── terraform/        # AWS VPC, MSK, ECS Fargate, ElastiCache, S3, IAM
│   └── docker/           # Prometheus + Grafana configs
├── notebooks/
│   ├── 01_graph_exploration.ipynb
│   └── 02_sim_swap_model.ipynb
└── tests/                # 140 tests — unit / integration / e2e
```

---

## Phases completed

| Phase | Component | Highlight |
|---|---|---|
| 1–2 | Scaffold + shared library | Pydantic schemas, SA validators, constants |
| 3 | Synthetic data generation | SA-locale Faker, PayShap IDs, load shedding injection |
| 4 | SIM swap model | LightGBM, 34 features, temporal CV — **AUC 0.9922** |
| 5 | GNN fraud ring detector | GraphSAGE on heterogeneous graph |
| 6 | Federated learning | DPFedAvg, ε-differential privacy, 5 SA bank shards |
| 7 | Streaming pipeline | Kafka → enrichment consumer → Redis velocity |
| 8 | FastAPI scoring layer | `/score`, `/explain`, `/debug-features` |
| 9 | Explainability | SHAP + GNN attribution + LLM narratives (EN + isiZulu) |
| 10 | Monitoring | PSI drift detector, Prometheus metrics, Grafana dashboard |
| 11 | AWS Terraform | VPC, MSK, ECS Fargate, ElastiCache, S3, IAM |
| 12 | Test suite | 140 tests — unit / integration / e2e, all passing |
| 13 | SIM swap notebook | Feature distributions, temporal CV, AUC/PR curves, SHAP waterfall |

---

## Quick start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Generate synthetic SA transaction data
python -m data_generation.scripts.generate_training_data

# 3. Train SIM swap model
python -m models.sim_swap.train

# 4. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Score a transaction
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

# 6. Run tests
pytest tests/
```

---

## SA domain features

**SIM swap detection** — SA's dominant fraud vector (60% of mobile banking breaches, ~R5.3B/year). Model features include swap-to-transaction timing, device fingerprint changes, OTP request patterns, and load shedding coincidence.

**Load shedding awareness** — Eskom rolling blackouts create legitimate connectivity gaps that mimic SIM swap indicators. The model cross-references outage schedules to distinguish infrastructure noise from genuine fraud signals. No international fraud model handles this.

**PayShap integration** — SARB's instant payment system (136M transactions, R100B+ since 2023) uses phone-number ShapIDs (+27XXXXXXXXX@bank), forming a natural bipartite graph for GNN-based mule network detection.

**POPIA Section 71** — Restricts purely automated decisions with significant effects. Every model decision ships with a SHAP explanation and an LLM-generated natural language narrative in English and isiZulu, satisfying the "sufficient information about underlying logic" requirement.

---

## Infrastructure

AWS-targeted (Capitec, TymeBank, Standard Bank all run on AWS):

| Component | Service |
|---|---|
| Event streaming | Amazon MSK (Kafka) |
| Model serving | ECS Fargate |
| Feature store | ElastiCache (Redis) |
| Artifact storage | S3 |
| Networking | VPC with public/private subnets |

---

## Models

| Model | Architecture | AUC |
|---|---|---|
| SIM swap detector | LightGBM, 34 features, temporal CV | **0.9922** |
| GNN fraud ring detector | GraphSAGE, heterogeneous graph | 0.6450 |
| Federated ensemble | DPFedAvg, 5 SA bank shards, ε-DP | — |

---

## Tech stack

`Python 3.11` · `LightGBM` · `PyTorch Geometric` · `Flower (federated)` · `FastAPI` · `Kafka (aiokafka)` · `Redis` · `SHAP` · `MLflow` · `Prometheus` · `Grafana` · `Terraform` · `Docker` · `pytest`

---

*Built by [Mmachelane Karabo Moswane](https://github.com/mmachelane) — Zaio Institute of Technology*
