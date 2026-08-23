"""The prediction API: one endpoint, calibrated intervals, hot-reload on promotion,
and a prediction log that feeds drift monitoring."""
from __future__ import annotations

import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import db
from .env import ROOT
from .registry import ProductionModel

PRED_DB = ROOT / "reports" / "predictions.db"

app = FastAPI(title="AppraiseNet", description="Calibrated used-vehicle price estimation.",
              version="1.0")
_model: ProductionModel | None = None


def model() -> ProductionModel:
    global _model
    if _model is None:
        _model = ProductionModel()
    elif _model.stale():
        _model.reload()
    return _model


class Listing(BaseModel):
    year: int = Field(ge=1980, le=2030)
    make: str
    model: str
    mileage: float = Field(gt=0)
    trim: str | None = None
    body_style: str | None = None
    drivetrain: str | None = None
    transmission: str | None = None
    fuel_type: str | None = None
    electrification: str | None = None
    gvwr_class: str | None = None
    series: str | None = None
    plant_country: str | None = None
    adaptive_cruise: str | None = None
    seller_type: str = "dealer"
    region_state: str | None = None
    cylinders: float | None = None
    doors: float | None = None
    engine_hp: float | None = None
    displacement_l: float | None = None
    original_price: float | None = None


@app.get("/health")
def health():
    m = model()
    return {"ok": True, "service": "AppraiseNet", "model_version": m.meta["version"]}


@app.post("/predict")
def predict(listing: Listing):
    t0 = time.time()
    m = model()
    row = listing.model_dump()
    row["age"] = max(2026 - row["year"], 0)
    row["miles_per_year"] = round(row["mileage"] / max(row["age"], 1))
    row["price"] = 0  # placeholder column for the feature space; never used as input
    try:
        out = m.predict(pd.DataFrame([row]))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out["ms"] = int((time.time() - t0) * 1000)
    _log(row, out)
    return out


def _log(row: dict, out: dict) -> None:
    db.log_prediction(row, out, sqlite_fallback=PRED_DB)
