import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import boto3
from google.cloud import storage
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Handles data ingestion from multiple sources."""

    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ingest_from_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Ingest data from CSV file."""
        logger.info(f"Ingesting CSV: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from CSV")
        return df

    def ingest_from_database(
        self,
        connection_string: str,
        query: str,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Ingest data from SQL database."""
        logger.info("Connecting to database...")
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        logger.info(f"Loaded {len(df)} rows from database")
        return df

    def ingest_from_s3(
        self,
        bucket: str,
        key: str,
        file_format: str = "parquet",
    ) -> pd.DataFrame:
        """Ingest data from AWS S3."""
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3 = boto3.client("s3")
        local_path = self.output_dir / Path(key).name
        s3.download_file(bucket, key, str(local_path))

        if file_format == "parquet":
            df = pd.read_parquet(local_path)
        elif file_format == "csv":
            df = pd.read_csv(local_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        logger.info(f"Loaded {len(df)} rows from S3")
        return df

    def ingest_from_gcs(
        self,
        bucket_name: str,
        blob_name: str,
        file_format: str = "parquet",
    ) -> pd.DataFrame:
        """Ingest data from Google Cloud Storage."""
        logger.info(f"Downloading gs://{bucket_name}/{blob_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_path = self.output_dir / Path(blob_name).name
        blob.download_to_filename(str(local_path))

        if file_format == "parquet":
            df = pd.read_parquet(local_path)
        else:
            df = pd.read_csv(local_path)

        logger.info(f"Loaded {len(df)} rows from GCS")
        return df

    def validate_schema(self, df: pd.DataFrame, schema: Dict[str, str]) -> bool:
        """Validate dataframe against expected schema."""
        errors = []
        for col, dtype in schema.items():
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
            elif str(df[col].dtype) != dtype:
                errors.append(f"Wrong dtype for {col}: expected {dtype}, got {df[col].dtype}")

        if errors:
            logger.error(f"Schema validation failed: {errors}")
            return False
        logger.info("Schema validation passed")
        return True

    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> str:
        """Save dataframe as parquet with metadata."""
        output_path = self.output_dir / filename
        table = pa.Table.from_pandas(df)
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "rows": str(len(df)),
            "columns": str(len(df.columns)),
        }
        table = table.replace_schema_metadata({**table.schema.metadata, **metadata})
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return str(output_path)


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    df = pd.DataFrame(np.random.randn(1000, 5), columns=list("ABCDE"))
    path = pipeline.save_to_parquet(df, "sample_data.parquet")
    print(f"Data saved to: {path}")
