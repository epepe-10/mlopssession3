"""Serve the model behind @staging as a REST API.
Usage: uvicorn app:app [--workers N]
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# the model lists its full training environment; the service only needs what is installed
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
log = logging.getLogger("uvicorn.error")

NAME = os.getenv("MODEL_NAME", "fraud-detector")
ALIAS = os.getenv("MODEL_ALIAS", "staging")
BUDGET_MS = 200

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the model, not on every request."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    state["model"] = mlflow.pyfunc.load_model(f"models:/{NAME}@{ALIAS}")
    state["version"] = mlflow.MlflowClient().get_model_version_by_alias(NAME, ALIAS).version
    log.info("loaded %s@%s = v%s", NAME, ALIAS, state["version"])
    yield
    state.clear()


app = FastAPI(title="Fraud scoring", lifespan=lifespan)


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    distance_km: float = Field(ge=0)
    merchant: Literal["grocery", "restaurant", "gas", "travel", "electronics", "crypto"]
    online: bool
    tx_last_24h: int = Field(ge=0)


class Prediction(BaseModel):
    fraud_proba: float
    model_version: str
    model_ms: float  # time spent scoring, for the latency budget


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    fraud_proba: list[float]
    model_version: str
    model_ms: float


def score(txs: list[Transaction]) -> list[float]:
    """Preprocessing lives HERE, so callers never have to know that the model wants online as 0/1."""
    X = pd.DataFrame([t.model_dump() for t in txs]).assign(online=lambda d: d.online.astype(int))
    return state["model"].predict(X)[:, 1].tolist()  # column 1 = P(fraud)


async def timed_score(txs: list[Transaction]) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    proba = await asyncio.to_thread(score, txs)  # keep the event loop free while the CPU works
    return proba, (time.perf_counter() - t0) * 1000


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    if ms > BUDGET_MS:
        log.warning("SLOW %s %s %d %.0f ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict")
async def predict(tx: Transaction) -> Prediction:
    proba, ms = await timed_score([tx])
    return Prediction(fraud_proba=proba[0], model_version=state["version"], model_ms=ms)


@app.post("/predict_batch")
async def predict_batch(batch: Batch) -> BatchPrediction:
    proba, ms = await timed_score(batch.transactions)
    return BatchPrediction(fraud_proba=proba, model_version=state["version"], model_ms=ms)
