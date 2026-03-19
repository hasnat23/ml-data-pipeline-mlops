import time
import logging
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Pipeline Serving API",
    description="Production API for ML model inference with monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
PIPELINE = None


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Input feature vector")
    model_version: Optional[str] = Field(default="latest", description="Model version")


class BatchPredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="Batch of feature vectors")


class PredictResponse(BaseModel):
    prediction: Any
    probability: Optional[List[float]] = None
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    global MODEL, PIPELINE
    model_path = Path("./artifacts/model.pkl")
    pipeline_path = Path("./artifacts/pipeline.pkl")

    if model_path.exists():
        MODEL = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning("No model found at startup")

    if pipeline_path.exists():
        PIPELINE = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    features = np.array(request.features).reshape(1, -1)

    if PIPELINE is not None:
        features = PIPELINE.transform(features)

    prediction = MODEL.predict(features)[0]
    probability = None
    if hasattr(MODEL, "predict_proba"):
        probability = MODEL.predict_proba(features)[0].tolist()

    latency_ms = (time.time() - start) * 1000

    return PredictResponse(
        prediction=prediction,
        probability=probability,
        model_version=request.model_version or "latest",
        latency_ms=round(latency_ms, 3),
    )


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    features = np.array(request.instances)

    if PIPELINE is not None:
        features = PIPELINE.transform(features)

    predictions = MODEL.predict(features).tolist()
    latency_ms = (time.time() - start) * 1000

    return {
        "predictions": predictions,
        "count": len(predictions),
        "latency_ms": round(latency_ms, 3),
    }


@app.get("/model/info")
async def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(MODEL).__name__,
        "has_predict_proba": hasattr(MODEL, "predict_proba"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
