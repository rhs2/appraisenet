"""The evaluation protocol: one honest procedure applied to every model.

Per model:
  1. K-fold cross-validation on the training partition. The `FeatureSpace` (categorical
     levels, imputation statistics, the trim-tier transform) is re-fitted on each fold's
     training rows, so no fitted statistic ever sees the rows it is evaluated on.
  2. Out-of-fold predictions give the selection metrics and the conformal calibration
     residuals.
  3. One final fit on the full training partition predicts the untouched holdout: the
     only numbers quoted as performance.
  4. Intervals: split-conformal in log space (the 80% quantile of |OOF residual| widens
     a symmetric band; guaranteed marginal coverage) for every model, plus conformalised
     quantile regression (CQR) for the production champion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import TARGET, Split
from .features import FeatureSpace
from .models import zoo


def metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict[str, float]:
    ape = np.abs(np.exp(p_log - y_log) - 1)
    ss = 1 - np.sum((y_log - p_log) ** 2) / np.sum((y_log - np.mean(y_log)) ** 2)
    return {"mape_pct": float(ape.mean() * 100), "median_ape_pct": float(np.median(ape) * 100),
            "r2_log": float(ss), "within_10_pct": float(np.mean(ape <= 0.10) * 100)}


@dataclass
class ModelResult:
    name: str
    family: str
    cv: dict[str, float]
    holdout: dict[str, float]
    interval: dict[str, float]
    fit_seconds: float
    oof: np.ndarray = field(repr=False, default=None)
    holdout_pred: np.ndarray = field(repr=False, default=None)

    def row(self) -> dict:
        return {"model": self.name, "family": self.family,
                **{f"cv_{k}": round(v, 3) for k, v in self.cv.items()},
                **{f"holdout_{k}": round(v, 3) for k, v in self.holdout.items()},
                **{k: round(v, 3) for k, v in self.interval.items()},
                "fit_seconds": round(self.fit_seconds, 1)}


def run_model(name: str, split: Split, cfg: Config) -> ModelResult:
    proto = cfg.protocol
    train, hold = split.train, split.holdout
    y = train[TARGET].values
    t0 = time.time()

    oof = np.zeros(len(train))
    for k in range(proto.folds):
        tr, ho = train[split.folds != k], train[split.folds == k]
        fs = FeatureSpace().fit(tr)
        oof[split.folds == k] = zoo.fit_predict(name, tr, ho, fs, cfg)

    fs_all = FeatureSpace().fit(train)
    hold_pred = zoo.fit_predict(name, train, hold, fs_all, cfg)
    fit_seconds = time.time() - t0

    resid = np.abs(y - oof)
    n = len(resid)
    q = float(np.quantile(resid, min(1.0, np.ceil((n + 1) * proto.target_coverage) / n)))
    yh = hold[TARGET].values
    cover = float(np.mean(np.abs(yh - hold_pred) <= q) * 100)
    width = float(np.median((np.exp(hold_pred + q) - np.exp(hold_pred - q)) / np.exp(yh)) * 100)
    interval = {"conformal_q_log": round(q, 4), "holdout_coverage_pct": round(cover, 1),
                "holdout_width_pct_of_price": round(width, 1)}

    return ModelResult(name=name, family=zoo.FAMILY[name], cv=metrics(y, oof),
                       holdout=metrics(yh, hold_pred), interval=interval,
                       fit_seconds=fit_seconds, oof=oof, holdout_pred=hold_pred)


def cqr_champion(split: Split, cfg: Config) -> dict[str, float]:
    """Conformalised quantile regression for the production champion (LightGBM)."""
    proto = cfg.protocol
    train, hold = split.train, split.holdout
    lo = np.zeros(len(train))
    hi = np.zeros(len(train))
    for k in range(proto.folds):
        tr = train[split.folds != k]
        fs = FeatureSpace().fit(tr)
        qm = zoo.quantile_models(tr, fs)
        frame = fs.tree_frame(train[split.folds == k])
        lo[split.folds == k] = qm["p10"].predict(frame)
        hi[split.folds == k] = qm["p90"].predict(frame)
    y = train[TARGET].values
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * proto.target_coverage) / n)))

    fs_all = FeatureSpace().fit(train)
    qm = zoo.quantile_models(train, fs_all)
    frame = fs_all.tree_frame(hold)
    tlo, thi = qm["p10"].predict(frame) - q, qm["p90"].predict(frame) + q
    yh = hold[TARGET].values
    return {"cqr_widen_log": round(q, 4),
            "cqr_holdout_coverage_pct": round(float(np.mean((yh >= tlo) & (yh <= thi)) * 100), 1),
            "cqr_holdout_width_pct_of_price": round(float(np.median((np.exp(thi) - np.exp(tlo)) / np.exp(yh)) * 100), 1)}


def segment_table(hold: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Holdout error by the slices where pricing models usually hide their weaknesses."""
    yh = hold[TARGET].values
    segs = {
        "under $10k": hold["price"] < 10_000,
        "$10k-20k": hold["price"].between(10_000, 20_000),
        "$20k-40k": hold["price"].between(20_000, 40_000),
        "over $40k": hold["price"] > 40_000,
        "age 0-3y": hold["age"] <= 3,
        "age 4-10y": hold["age"].between(4, 10),
        "age 11y+": hold["age"] >= 11,
        "150k+ miles": hold["mileage"] >= 150_000,
        "private seller": hold["seller_type"] == "private",
    }
    rows = []
    for seg, mask in segs.items():
        m = mask.values
        if m.sum() < 20:
            continue
        row = {"segment": seg, "n": int(m.sum())}
        for name, p in preds.items():
            row[name] = round(float(np.mean(np.abs(np.exp(p[m] - yh[m]) - 1)) * 100), 2)
        rows.append(row)
    return pd.DataFrame(rows)
