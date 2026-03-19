import logging
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter("model_predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram("model_prediction_latency_seconds", "Prediction latency")
MODEL_ACCURACY = Gauge("model_accuracy", "Current model accuracy")
DATA_DRIFT_SCORE = Gauge("data_drift_score", "Data drift detection score")


class ModelMonitor:
    """Real-time model monitoring with drift detection."""

    def __init__(self, reference_data: pd.DataFrame, model=None, alert_threshold: float = 0.05):
        self.reference_data = reference_data
        self.model = model
        self.alert_threshold = alert_threshold
        self.predictions_log = []
        self.performance_history = []

    def detect_data_drift(
        self, current_data: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict:
        """Detect data drift using KS test."""
        if columns is None:
            columns = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()

        drift_results = {}
        drifted_features = []

        for col in columns:
            if col not in self.reference_data.columns or col not in current_data.columns:
                continue

            ref = self.reference_data[col].dropna().values
            cur = current_data[col].dropna().values

            ks_stat, p_value = stats.ks_2samp(ref, cur)
            is_drifted = p_value < self.alert_threshold

            drift_results[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "is_drifted": is_drifted,
            }

            if is_drifted:
                drifted_features.append(col)

        drift_score = len(drifted_features) / max(len(columns), 1)
        DATA_DRIFT_SCORE.set(drift_score)

        summary = {
            "drift_score": drift_score,
            "drifted_features": drifted_features,
            "total_features": len(columns),
            "feature_drift": drift_results,
            "drift_detected": drift_score > 0.3,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if summary["drift_detected"]:
            logger.warning(f"DATA DRIFT DETECTED! Score: {drift_score:.2f}, Features: {drifted_features}")
        else:
            logger.info(f"No significant drift. Score: {drift_score:.2f}")

        return summary

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: any,
        ground_truth: Optional[any] = None,
        latency: float = 0.0,
    ):
        """Log individual prediction for monitoring."""
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(latency)

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "ground_truth": ground_truth,
            "latency": latency,
        }
        self.predictions_log.append(record)

    def compute_performance_metrics(
        self, y_true: List, y_pred: List, window: int = 1000
    ) -> Dict:
        """Compute performance metrics over a sliding window."""
        if len(y_true) < window:
            window = len(y_true)

        y_true_window = y_true[-window:]
        y_pred_window = y_pred[-window:]

        metrics = {
            "accuracy": accuracy_score(y_true_window, y_pred_window),
            "f1": f1_score(y_true_window, y_pred_window, average="weighted"),
            "window_size": window,
            "timestamp": datetime.utcnow().isoformat(),
        }

        MODEL_ACCURACY.set(metrics["accuracy"])
        self.performance_history.append(metrics)
        logger.info(f"Performance metrics: accuracy={metrics['accuracy']:.4f}")
        return metrics

    def generate_report(self) -> Dict:
        """Generate monitoring report."""
        return {
            "total_predictions": len(self.predictions_log),
            "performance_history": self.performance_history[-10:],
            "generated_at": datetime.utcnow().isoformat(),
        }

    def start_metrics_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
