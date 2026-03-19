import argparse
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
}


class ModelTrainer:
    """MLOps-ready model training with MLflow tracking."""

    def __init__(self, experiment_name: str = "ml-pipeline", tracking_uri: str = "./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.best_model = None
        self.best_metrics = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = "random_forest",
        hyperparams: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
    ):
        """Train model with MLflow run tracking."""
        if model_type not in MODELS:
            raise ValueError(f"Unknown model: {model_type}. Choose from {list(MODELS.keys())}")

        hyperparams = hyperparams or {}
        model_cls = MODELS[model_type]
        model = model_cls(**hyperparams)

        with mlflow.start_run(run_name=run_name or model_type):
            mlflow.log_param("model_type", model_type)
            mlflow.log_params(hyperparams)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))

            logger.info(f"Training {model_type}...")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "f1": f1_score(y_val, y_pred, average="weighted"),
                "precision": precision_score(y_val, y_pred, average="weighted"),
                "recall": recall_score(y_val, y_pred, average="weighted"),
            }

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y_val, y_prob[:, 1])

            mlflow.log_metrics(metrics)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())

            # Log model
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id

        logger.info(f"Training complete. Metrics: {metrics}")
        self.best_model = model
        self.best_metrics = metrics
        return model, metrics, run_id

    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        param_grid: Dict[str, Any],
        cv: int = 5,
    ):
        """Grid search for hyperparameter optimization."""
        model_cls = MODELS[model_type]
        model = model_cls()
        search = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

        with mlflow.start_run(run_name=f"{model_type}_grid_search"):
            search.fit(X_train, y_train)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_score", search.best_score_)
            mlflow.sklearn.log_model(search.best_estimator_, "best_model")

        logger.info(f"Best params: {search.best_params_}, Score: {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_

    def save_model(self, model, path: str):
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ModelTrainer(experiment_name="demo-experiment")
    model, metrics, run_id = trainer.train(
        X_train, y_train, X_val, y_val,
        model_type="random_forest",
        hyperparams={"n_estimators": 100, "max_depth": 10, "random_state": 42},
    )
    print(f"Run ID: {run_id}")
    print(f"Metrics: {metrics}")
