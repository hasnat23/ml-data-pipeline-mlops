import logging
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
import joblib

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Automated feature engineering for ML pipelines."""

    def __init__(self):
        self.transformers = {}
        self.feature_names = []
        self.pipeline = None

    def detect_column_types(
        self, df: pd.DataFrame, target_col: str = None
    ) -> Dict[str, List[str]]:
        """Auto-detect numerical and categorical columns."""
        cols = [c for c in df.columns if c != target_col]
        numerical = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical = df[cols].select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df[cols].select_dtypes(include=["datetime"]).columns.tolist()
        return {"numerical": numerical, "categorical": categorical, "datetime": datetime_cols}

    def create_time_features(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Extract time-based features from datetime column."""
        df = df.copy()
        dt = pd.to_datetime(df[datetime_col])
        df[f"{datetime_col}_year"] = dt.dt.year
        df[f"{datetime_col}_month"] = dt.dt.month
        df[f"{datetime_col}_day"] = dt.dt.day
        df[f"{datetime_col}_dayofweek"] = dt.dt.dayofweek
        df[f"{datetime_col}_hour"] = dt.dt.hour
        df[f"{datetime_col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        df[f"{datetime_col}_quarter"] = dt.dt.quarter
        logger.info(f"Created time features from {datetime_col}")
        return df

    def create_interaction_features(
        self, df: pd.DataFrame, col_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Create interaction features between column pairs."""
        df = df.copy()
        for col1, col2 in col_pairs:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
            df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
        logger.info(f"Created {len(col_pairs) * 3} interaction features")
        return df

    def build_preprocessing_pipeline(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        scaling: str = "standard",
    ) -> ColumnTransformer:
        """Build sklearn preprocessing pipeline."""
        scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.pipeline = ColumnTransformer([
            ("numerical", numerical_pipeline, numerical_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ])

        logger.info("Preprocessing pipeline built")
        return self.pipeline

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """Fit and transform the dataset."""
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None

        X_transformed = self.pipeline.fit_transform(X)
        logger.info(f"Transformed dataset: {X.shape} -> {X_transformed.shape}")
        return X_transformed, y

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 20,
        task: str = "classification",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top k features using statistical tests."""
        score_func = f_classif if task == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        mask = selector.get_support()
        logger.info(f"Selected {k} features from {X.shape[1]}")
        return X_selected, mask

    def save_pipeline(self, path: str):
        """Save the preprocessing pipeline."""
        joblib.dump(self.pipeline, path)
        logger.info(f"Pipeline saved to {path}")

    def load_pipeline(self, path: str):
        """Load a previously saved pipeline."""
        self.pipeline = joblib.load(path)
        logger.info(f"Pipeline loaded from {path}")
        return self.pipeline
