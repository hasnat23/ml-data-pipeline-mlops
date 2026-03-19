# ML Data Pipeline & MLOps

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.x-blue.svg)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.7-red.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A **production-grade MLOps pipeline** for automated machine learning workflows. Covers end-to-end ML lifecycle management: data ingestion, feature engineering, model training, evaluation, deployment, and monitoring with drift detection and A/B testing.

## Architecture Overview

```
Data Sources → Ingestion → Feature Store → Training → Model Registry → Deployment
                                                              ↓
                                                       Monitoring & Drift Detection
```

## Key Features

- **Automated Data Pipelines** with Apache Airflow DAGs
- **Feature Store** using Feast for consistent feature serving
- **Experiment Tracking** with MLflow (metrics, params, artifacts)
- **Model Registry** with automated versioning and staging
- **CI/CD for ML** with GitHub Actions + automated testing
- **Model Monitoring** with Evidently AI for drift detection
- **A/B Testing** framework for model comparison in production
- **Containerized** with Docker Compose for reproducibility

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | Apache Airflow 2.7 |
| Experiment Tracking | MLflow 2.x |
| Feature Store | Feast |
| Model Serving | FastAPI + BentoML |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Data Processing | Pandas, PySpark, Great Expectations |
| Storage | PostgreSQL, MinIO (S3-compatible) |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## Project Structure

```
ml-data-pipeline-mlops/
├── dags/
│   ├── data_ingestion_dag.py     # Airflow DAG for data collection
│   ├── feature_engineering_dag.py # Feature computation pipeline
│   ├── training_dag.py           # Automated model training
│   └── deployment_dag.py         # Model deployment pipeline
├── src/
│   ├── data/
│   │   ├── ingestion.py           # Data collection & validation
│   │   └── preprocessing.py       # Data cleaning & transformation
│   ├── features/
│   │   ├── feature_store.py       # Feast feature definitions
│   │   └── feature_engineering.py # Feature computation logic
│   ├── training/
│   │   ├── trainer.py             # Model training with MLflow
│   │   └── hyperopt.py            # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── metrics.py             # Model evaluation metrics
│   │   └── validation.py          # Model validation checks
│   ├── serving/
│   │   ├── api.py                 # FastAPI model serving
│   │   └── ab_testing.py          # A/B testing logic
│   └── monitoring/
│       ├── drift_detector.py      # Evidently AI drift detection
│       └── alerts.py              # Alerting & notifications
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_feature_engineering.py
│   └── test_model_serving.py
├── docker-compose.yml
├── .github/workflows/
│   └── ml_ci_cd.yml              # GitHub Actions CI/CD
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and start all services
git clone https://github.com/hasnat23/ml-data-pipeline-mlops.git
cd ml-data-pipeline-mlops
docker-compose up -d

# Access services
# Airflow UI:  http://localhost:8080
# MLflow UI:   http://localhost:5000
# Grafana:     http://localhost:3000
# FastAPI:     http://localhost:8000/docs

# Trigger training pipeline
python src/training/trainer.py --config configs/training_config.yaml
```

## MLflow Experiment Tracking

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"model": "XGBoost", "n_estimators": 100})
    mlflow.log_metrics({"accuracy": 0.94, "f1": 0.93, "auc": 0.97})
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("stage", "production-candidate")
```

## Model Performance Dashboard

| Model | Accuracy | F1 | AUC | Latency (ms) |
|---|---|---|---|---|
| XGBoost v1 | 91.2% | 0.908 | 0.961 | 12ms |
| LightGBM v2 | 93.7% | 0.934 | 0.971 | 8ms |
| Neural Net v1 | 94.1% | 0.938 | 0.975 | 45ms |

## Author

**Muhammad Hasnat**  
ML & AI Engineer | MLOps Specialist  
[LinkedIn](https://linkedin.com/in/hasnat23) | [GitHub](https://github.com/hasnat23)

## License

MIT License
