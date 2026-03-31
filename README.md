<div align="center">

# OptiCloud AI

### Real-Time Cloud Cost Intelligence & Adaptive Optimization System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20Lambda%20%7C%20S3-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/ML-HistGradientBoosting-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Author](https://img.shields.io/badge/Author-Hemang%20Agarwal-0E7C7B?style=flat-square)](.)

> **"Cloud systems today are good at producing data — but poor at producing decisions."**
> OptiCloud AI bridges that gap.

</div>

---

## The Problem

Organizations running cloud workloads face a compounding, silent problem: **uncontrolled spending driven by waste they can't see in time.**

| Problem | Reality |
|---|---|
| **Reactive monitoring** | Teams discover cost spikes *after* budget is already gone |
| **Generic alerting** | A CPU spike alone doesn't mean a real problem — simple thresholds create noise, not insight |
| **No intelligent action** | Even when anomalies are flagged, the system offers no decision — engineers must investigate manually |

OptiCloud AI addresses all three. It doesn't just detect anomalies — it **scores their cost impact, decides the appropriate action, and executes it safely** based on how confident it is.

---

## How It Works

```
Live AWS Metrics (EC2 / Lambda / S3)
          │
          ▼
  collector_realtime.py        ← Pulls CloudWatch metrics via Boto3
          │
          ▼
  preprocess_realtime.py       ← Rolling stats, pct_change, feature engineering
          │
          ▼
  damp_stage.py                ← Anomaly scoring (pattern-based, not threshold-based)
          │
          ▼
  lightgbm_stage.py            ← Trains HistGradientBoosting on rule-assisted labels
          │
          ▼
  decision_stage_v3.py         ← Confidence scoring + cost estimation + action selection
          │
          ▼
  dashboard/app.py             ← Streamlit dashboard: anomalies, actions, before/after cost
```

---

## Infrastructure

| Component | Count | Purpose |
|---|---|---|
| EC2 Instances | 13 (incl. monitoring node) | Distributed compute workloads |
| AWS Lambda Functions | 5 | Event-driven processing |
| S3 Buckets | 2 | Raw metrics + processed outputs |
| CloudWatch | — | Live metric source |

### S3 Bucket Layout

```
sourav-cost-low-20260328     ← Raw: CloudWatch snapshots, metric JSON files
sourav-cost-high-20260328    ← Processed: features, anomaly scores, final_output.csv
```

---

## Setup

### Prerequisites

- Python 3.9+
- AWS account (Free Tier compatible)
- AWS CLI installed

### Step 1 — Install AWS CLI (Windows)

```
https://awscli.amazonaws.com/AWSCLIV2.msi
```

Verify:
```bash
aws --version
```

### Step 2 — Create IAM User

In **AWS Console → IAM → Users → Create User** with Programmatic Access. Attach:

- `AmazonEC2FullAccess`
- `AmazonS3FullAccess`
- `AWSLambdaFullAccess`
- `CloudWatchReadOnlyAccess`

### Step 3 — Configure CLI

```bash
aws configure
```

```
AWS Access Key ID:      <your-key>
AWS Secret Access Key:  <your-secret>
Default region:         ap-south-1
Output format:          json
```

### Step 4 — Create S3 Buckets

```bash
aws s3 mb s3://sourav-cost-low-20260328
aws s3 mb s3://sourav-cost-high-20260328
```

>  **Free Tier:** EC2 gives 750 hrs/month for `t2.micro`/`t3.micro`. Running all 13 instances simultaneously will exceed free limits. Use 2–3 instances during local testing.

---

## Running the Project

### 1. Set Up Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements-pipeline.txt
pip install -r requirements-dashboard.txt
```

### 2. Single Pipeline Run

Executes one full cycle: collect → preprocess → score → decide.

```bash
python scripts/run_pipeline.py
```

### 3. Continuous Real-Time Loop

Runs the pipeline every 60 seconds. S3 collection runs every N cycles (default: 60), configurable via environment variable.

```bash
python scripts/run_pipeline_loop.py

# Optional: change S3 collection frequency
S3_EVERY_N_CYCLES=30 python scripts/run_pipeline_loop.py
```

### 4. Train the ML Model (run once before v3 decisions)

```bash
python scripts/lightgbm_stage.py
```

### 5. Launch Dashboard

## Pipeline — Stage by Stage

### `collector_realtime.py`

Pulls live metrics from AWS via CloudWatch and Boto3:

- **EC2:** CPU utilization, NetworkIn, NetworkOut — 20-minute lookback, 60s period
- **Lambda:** Invocations, Errors, Duration — 24-hour lookback (Lambda metrics are sparse/bursty)
- **S3:** BucketSizeBytes, NumberOfObjects — 2-day lookback (daily cadence)

Resources with no recent CloudWatch datapoints receive a `NoData` fallback row so they still appear in downstream stages rather than silently dropping out.

Use `--skip-s3` for fast minute-level loops:

```bash
python scripts/collector_realtime.py --skip-s3
```

---

### `preprocess_realtime.py`

Builds features per `(resource_type, resource_id, metric_name)` group:

| Feature | Description |
|---|---|
| `rolling_mean_3` | 3-point rolling mean |
| `rolling_std_3` | 3-point rolling standard deviation |
| `pct_change` | Percentage change between consecutive readings |

Outputs: `features_<timestamp>.csv` + `.parquet` → uploaded to `sourav-cost-high-20260328/processed/`

---

### `damp_stage.py`

Computes anomaly scores without requiring labeled data:

```
anomaly_score = 0.5 × |pct_change| + 0.5 × (rolling_std / rolling_mean)
```

Service-aware thresholds:

| Service | Threshold | Reason |
|---|---|---|
| EC2 | 0.8 | Avoid over-alerting on normal compute variation |
| Lambda | 0.4 | Serverless anomalies emerge earlier and need tighter detection |
| S3 | 0.8 | Storage metrics change slowly |

Two noise guards are applied: EC2 network metrics that drop to zero have their `pct_change` contribution dampened by 80%. Near-zero traffic baselines that would cause coefficient-of-variation explosion are also suppressed.

Outputs: `anomaly_scores_<timestamp>.csv` + `.parquet`

---

### `lightgbm_stage.py`

Trains a `HistGradientBoostingClassifier` on rule-assisted labels bootstrapped from the anomaly scores — no manually labeled training data required.

**Features used:**

```
value, anomaly_score, rolling_mean_3, rolling_std_3,
pct_change, is_ec2, is_lambda, is_s3, cost
```

**Action labels the model learns to predict:**

| Label | Trigger |
|---|---|
| `scale_down` | EC2 CPU < 15% + anomaly detected |
| `investigate_high_compute` | EC2 CPU > 75% |
| `investigate_network` | EC2 NetworkIn/Out or Disk anomaly |
| `optimize_lambda` | Lambda Duration anomaly |
| `limit_lambda` | Lambda Invocations anomaly |
| `investigate_lambda_errors` | Lambda Errors > 0 |
| `optimize_s3_lifecycle` | S3 BucketSizeBytes anomaly score ≥ 0.8 |
| `investigate_s3_access` | Other S3 anomalies |
| `monitor` | Score < 0.3 — normal, no action needed |

Saves model artifacts to `models/model.joblib` and `models/labels.joblib`.

---

### `decision_stage_v3.py`

The final decision layer. For each resource, it:

1. Loads the trained model and predicts the recommended action
2. Enforces a **service-action policy** — prevents cross-service mismatches (e.g. `scale_down` will never apply to a Lambda function)
3. Computes a **confidence score** from anomaly score + severity + metric type
4. Estimates **current cost** and **post-action cost** via the cost model
5. Determines **action status** from confidence

**Confidence → Action Status:**

| Confidence | Status |
|---|---|
| ≥ 0.95 | `executed` |
| ≥ 0.85 | `auto_fix_with_rollback` |
| ≥ 0.70 | `recommendation_with_approval` |
| < 0.70 | `monitor_only` |

**Cost reduction multipliers** (applied only for optimization actions):

| Action | Cost After |
|---|---|
| `scale_down` | 50% of before |
| `optimize_lambda` | 60% of before |
| `limit_lambda` | 70% of before |
| `optimize_s3_lifecycle` | 75% of before |

Investigate and monitor actions never modify cost estimates.

Outputs: `data/processed/final_output.csv` → uploaded to `sourav-cost-high-20260328/processed/final_output.csv`

---

### `cost_model.py`

Shared pricing and policy module used across the pipeline:

| Service | Pricing Basis |
|---|---|
| EC2 `t3.micro` | $0.0104/hour |
| Lambda requests | $0.20 per 1M requests |
| Lambda compute | $0.0000166667 per GB-second |
| S3 Standard | $0.023 per GB-month |

---

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) reads `final_output.csv` and displays:

- **Cost comparison** — estimated spend before vs. after optimization per resource
- **Anomaly map** — which instances/functions triggered detection and at what score
- **Decision log** — every action taken, its confidence score, severity, and reasoning
- **Savings summary** — total estimated savings across the current cycle

---

## Project Structure

```
opticloud-ai/
├── scripts/
│   ├── collector_realtime.py       # AWS metric collection (EC2, Lambda, S3)
│   ├── preprocess_realtime.py      # Feature engineering (rolling stats, pct_change)
│   ├── damp_stage.py               # Anomaly scoring with service-aware thresholds
│   ├── lightgbm_stage.py           # ML model training (HistGradientBoosting)
│   ├── decision_stage_v3.py        # Cost estimation + action decision + confidence
│   ├── cost_model.py               # Shared pricing + action policy logic
│   ├── check_labels.py             # Utility: inspect label class distribution
│   ├── run_pipeline.py             # Single full pipeline execution
│   └── run_pipeline_loop.py        # Continuous 60s loop with S3 cycle control
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── data/
│   ├── raw/                        # metrics_<timestamp>.json from collector
│   └── processed/                  # features, anomaly_scores, final_output.csv
├── models/
│   ├── model.joblib                # Trained HistGradientBoosting classifier
│   └── labels.joblib               # Inverse label map (int → action string)
├── requirements-pipeline.txt
├── requirements-dashboard.txt
└── README.md
```

---

## Key Design Decisions

**Why pattern-based anomaly scoring over thresholds?**
A single high CPU reading could be a normal scheduled job. The scoring formula combines rate-of-change and coefficient-of-variation so the system detects *how unusual a pattern is relative to recent history*, not just whether a value crossed a line.

**Why rule-assisted label bootstrapping for the ML model?**
There is no labeled cloud cost dataset available for this environment. Rather than treating this as purely unsupervised, the system generates high-quality labels from domain knowledge rules and uses them to train a classifier that generalizes to unseen metric combinations — giving the best of both approaches.

**Why two separate decision stages?**
`decision_stage_v3.py` uses the trained ML model and is the default. Both share the same cost model and service-action policy layer so outputs are consistent.

**Why is S3 collected less frequently?**
S3 storage metrics are daily snapshots in CloudWatch. Polling them every 60 seconds wastes API quota. The loop skips S3 by default and only includes it every N cycles, configurable via `S3_EVERY_N_CYCLES`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Cloud | AWS EC2, Lambda, S3, CloudWatch |
| AWS SDK | Boto3 |
| Anomaly Detection | Statistical (rolling CV + pct_change) |
| ML Classifier | `HistGradientBoostingClassifier` (scikit-learn) |
| Feature Engineering | pandas, numpy |
| Dashboard | Streamlit |
| Model Persistence | joblib |
| Data Storage | Amazon S3 (Parquet + CSV) |

---
<img width="1903" height="812" alt="image" src="https://github.com/user-attachments/assets/f5476013-8c81-4833-97db-538ab7ddee16" />
<img width="1903" height="812" alt="Screenshot 2026-03-31 225407" src="https://github.com/user-attachments/assets/62061110-7d4c-458c-82ef-b1c96da6d245" />
<img width="1901" height="602" alt="Screenshot 2026-03-31 225456" src="https://github.com/user-attachments/assets/69a06751-f637-49eb-b2e5-8f4ee72f3df2" />
<img width="1424" height="683" alt="image" src="https://github.com/user-attachments/assets/f9762777-521f-4c13-bbdc-80521fb72df2" />


## ⚠️ Notes

- Run `aws configure` before executing any script
- Both S3 buckets must exist before the pipeline starts
- Train the model (`lightgbm_stage.py`) at least once before running `decision_stage_v3.py`
- All generated files under `data/` and `models/` should be added to `.gitignore`
- The system targets non-critical workload reduction only — investigate/monitor actions never modify infrastructure

---

*Built by **Hemang Agarwal***
