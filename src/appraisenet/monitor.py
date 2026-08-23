"""Drift monitoring: population-stability index between the training reference and the
recent prediction traffic logged by serve.py (Postgres, or reports/predictions.db)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import db
from .config import Config
from .data import engineer, load_listings
from .env import ROOT

PRED_DB = ROOT / "reports" / "predictions.db"
NUM_WATCH = ["mileage", "age", "engine_hp"]
CAT_WATCH = ["make", "body_style", "seller_type", "region_state"]
PSI_WARN, PSI_ALERT = 0.10, 0.25


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref.dropna(), bins=edges)[0] / max(len(ref.dropna()), 1)
    c = np.histogram(cur.dropna(), bins=edges)[0] / max(len(cur.dropna()), 1)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series, top: int = 20) -> float:
    levels = ref.value_counts().head(top).index
    r = ref.value_counts(normalize=True).reindex(levels).fillna(1e-4)
    c = cur.value_counts(normalize=True).reindex(levels).fillna(1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def report(cfg: Config, hours: int = 24 * 7) -> dict:
    ref, _ = load_listings()
    ref = engineer(ref, cfg.protocol)
    rows = db.recent_predictions(hours, sqlite_fallback=PRED_DB)
    if not rows:
        return {"ok": False, "reason": "no prediction traffic logged yet"}
    if len(rows) < 30:
        return {"ok": False, "reason": f"only {len(rows)} predictions in the window; need 30+"}
    cur = pd.DataFrame([json.loads(r[0]) for r in rows])
    cur["pred_price"] = [r[1] for r in rows]

    out: dict = {"ok": True, "window_hours": hours, "n": len(cur), "features": {}}
    worst = 0.0
    for c in NUM_WATCH:
        if c in cur:
            v = psi_numeric(ref[c], pd.to_numeric(cur[c], errors="coerce"))
            out["features"][c] = round(v, 3)
            worst = max(worst, v)
    for c in CAT_WATCH:
        if c in cur:
            v = psi_categorical(ref[c].astype(str), cur[c].astype(str))
            out["features"][c] = round(v, 3)
            worst = max(worst, v)
    out["prediction_psi"] = round(psi_numeric(ref["price"], cur["pred_price"]), 3)
    worst = max(worst, out["prediction_psi"])
    out["status"] = "alert" if worst >= PSI_ALERT else "warn" if worst >= PSI_WARN else "healthy"
    return out
