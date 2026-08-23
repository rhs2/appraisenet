"""Production model registry: fit, version, promote-or-rollback.

`appraisenet train-production` fits the production configuration (LightGBM point model +
p10/p90 quantile models + conformal calibration + the fitted FeatureSpace) and stages it.
Promotion applies the guardrail: the candidate must match the serving model's holdout
MAPE within a tolerance, otherwise the serving model stays and the candidate is archived.
Versions are semantic: automatic retrains bump MINOR; MAJOR is a manual architecture
decision; the API hot-reloads on promotion.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np

from .config import Config
from .data import TARGET, engineer, load_listings, protocol_split
from .env import ROOT
from .features import FeatureSpace
from .models import zoo
from .protocol import metrics

CURRENT = ROOT / "models" / "current"
ARCHIVE = ROOT / "models" / "archive"
TOLERANCE = 0.30


def train_production(cfg: Config, force: bool = False) -> dict:
    df, synthetic = load_listings(max_rows=cfg.max_rows, seed=cfg.protocol.seed)
    df = engineer(df, cfg.protocol)
    split = protocol_split(df, cfg.protocol)
    train, hold = split.train, split.holdout
    y = train[TARGET].values

    # out-of-fold predictions -> conformal widths (models never score their own rows)
    oof = np.zeros(len(train))
    lo = np.zeros(len(train))
    hi = np.zeros(len(train))
    for k in range(cfg.protocol.folds):
        tr, ho = train[split.folds != k], train[split.folds == k]
        fs = FeatureSpace().fit(tr)
        oof[split.folds == k] = zoo.fit_predict("lightgbm", tr, ho, fs, cfg)
        qm = zoo.quantile_models(tr, fs)
        frame = fs.tree_frame(ho)
        lo[split.folds == k] = qm["p10"].predict(frame)
        hi[split.folds == k] = qm["p90"].predict(frame)
    n = len(y)
    tc = cfg.protocol.target_coverage
    cqr_q = float(np.quantile(np.maximum(lo - y, y - hi), min(1.0, np.ceil((n + 1) * tc) / n)))

    # candidate metrics on the untouched holdout (trained on the training partition only)
    fs_train = FeatureSpace().fit(train)
    hold_pred = zoo.fit_predict("lightgbm", train, hold, fs_train, cfg)
    cand = metrics(hold[TARGET].values, hold_pred)

    serving = current_meta()
    old_mape = serving.get("holdout", {}).get("mape_pct", float("inf"))
    if not force and cand["mape_pct"] > old_mape + TOLERANCE:
        return {"decision": "rejected", "candidate": cand, "serving": serving.get("holdout"),
                "reason": f"candidate MAPE {cand['mape_pct']:.2f}% vs serving {old_mape:.2f}% (+{TOLERANCE} tolerance)"}

    # production fit on ALL rows (the honest metrics above stay those of the train-only fit)
    import lightgbm as lgb
    fs_all = FeatureSpace().fit(df)
    point = lgb.train(zoo._lgbm_params(), lgb.Dataset(fs_all.tree_frame(df), df[TARGET].values),
                      num_boost_round=1500)
    qm = zoo.quantile_models(df, fs_all)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    if CURRENT.exists():
        dst = ARCHIVE / stamp
        dst.mkdir(parents=True, exist_ok=True)
        for f in CURRENT.iterdir():
            shutil.copy2(f, dst / f.name)
    CURRENT.mkdir(parents=True, exist_ok=True)
    point.save_model(str(CURRENT / "point.txt"))
    qm["p10"].save_model(str(CURRENT / "p10.txt"))
    qm["p90"].save_model(str(CURRENT / "p90.txt"))
    joblib.dump(fs_all, CURRENT / "feature_space.joblib")
    if serving.get("version"):
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(serving["version"]))
        version = f"{m.group(1)}.{int(m.group(2)) + 1}.0" if m else "1.0.0"
    else:
        version = "1.0.0"
    meta = {"version": version, "trained": stamp,
            "rows": len(df), "synthetic": bool(synthetic), "holdout": cand,
            "cqr_widen_log": round(cqr_q, 4), "target_coverage": tc,
            "decision": "promoted"}
    (CURRENT / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def current_meta() -> dict:
    p = CURRENT / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


class ProductionModel:
    """Loaded serving artifacts with hot-reload on promotion."""

    def __init__(self):
        self._mtime = 0.0
        self.reload()

    def stale(self) -> bool:
        p = CURRENT / "meta.json"
        return p.exists() and p.stat().st_mtime != self._mtime

    def reload(self) -> None:
        import lightgbm as lgb
        if not (CURRENT / "meta.json").exists():
            raise FileNotFoundError("no production model; run `appraisenet train-production` first")
        self.meta = current_meta()
        self.point = lgb.Booster(model_file=str(CURRENT / "point.txt"))
        self.p10 = lgb.Booster(model_file=str(CURRENT / "p10.txt"))
        self.p90 = lgb.Booster(model_file=str(CURRENT / "p90.txt"))
        self.fs: FeatureSpace = joblib.load(CURRENT / "feature_space.joblib")
        self._mtime = (CURRENT / "meta.json").stat().st_mtime

    def predict(self, frame) -> dict:
        X = self.fs.tree_frame(frame)
        q = self.meta["cqr_widen_log"]
        price = float(np.exp(self.point.predict(X)[0]))
        low = float(np.exp(self.p10.predict(X)[0] - q))
        high = float(np.exp(self.p90.predict(X)[0] + q))
        return {"price": round(price), "low": round(min(low, price)), "high": round(max(high, price)),
                "coverage": self.meta["target_coverage"], "model_version": self.meta["version"]}
