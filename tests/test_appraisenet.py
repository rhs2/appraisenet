"""Test suite. Everything runs on the synthetic corpus: CI never sees the private data."""
from __future__ import annotations

import numpy as np
import pytest

from appraisenet.config import load_config
from appraisenet.data import TARGET, engineer, load_listings, protocol_split, synthetic_listings
from appraisenet.features import FeatureSpace, TrimTier
from appraisenet.models import zoo
from appraisenet.protocol import metrics, run_model


@pytest.fixture(scope="session")
def cfg():
    c = load_config(None)
    c.protocol.folds = 3
    c.torch_epochs = 5
    return c


@pytest.fixture(scope="session")
def split(cfg):
    df = engineer(synthetic_listings(1200, seed=3), cfg.protocol)
    return protocol_split(df, cfg.protocol)


def test_synthetic_schema(cfg):
    df = synthetic_listings(200)
    for col in ["price", "year", "make", "model", "mileage", "seller_type", "description"]:
        assert col in df.columns
    e = engineer(df, cfg.protocol)
    assert e["price"].between(cfg.protocol.price_min, cfg.protocol.price_max).all()
    assert (e["year"] >= cfg.protocol.year_min).all()
    assert TARGET in e


def test_split_is_disjoint(split):
    assert len(set(split.train["id"]) & set(split.holdout["id"])) == 0
    assert len(split.folds) == len(split.train)


def test_feature_space_fits_on_train_only(split):
    fs = FeatureSpace().fit(split.train)
    unseen = set(split.holdout["make"].astype(str)) - set(fs.levels["make"])
    X = fs.tree_frame(split.holdout)
    assert X.shape[0] == len(split.holdout)
    # unseen categories become NaN codes, never new levels
    assert set(X["make"].cat.categories) == set(fs.levels["make"]) or not unseen


def test_trim_tier_never_leaks(split):
    tt = TrimTier().fit(split.train)
    got = tt.transform(split.holdout)
    assert set(got.unique()) <= {"top", "upper", "mid", "base", "unknown"}


def test_dense_and_onehot_finite(split):
    fs = FeatureSpace().fit(split.train)
    assert np.isfinite(fs.dense(split.holdout)).all()
    assert np.isfinite(fs.onehot(split.holdout)).all()
    cats, nums = fs.embed_tensors(split.holdout)
    assert np.isfinite(nums).all() and cats.min() >= 0


def test_metrics_shape():
    y = np.log(np.array([10000.0, 20000.0, 30000.0]))
    m = metrics(y, y)
    assert m["mape_pct"] == 0 and m["within_10_pct"] == 100


@pytest.mark.parametrize("name", ["ridge", "lightgbm"])
def test_run_model_end_to_end(name, split, cfg):
    res = run_model(name, split, cfg)
    assert 0 < res.holdout["mape_pct"] < 60
    assert 50 <= res.interval["holdout_coverage_pct"] <= 100
    assert len(res.oof) == len(split.train)


def test_zoo_registry_complete():
    assert set(zoo.ZOO) == set(zoo.FAMILY)


def test_paired_bootstrap_comparison(tmp_path):
    import pandas as pd

    from appraisenet.compare import comparison_stats
    rng = np.random.RandomState(0)
    y = np.log(rng.uniform(5000, 40000, 400))
    (tmp_path / "predictions").mkdir()
    preds = {"good": y + rng.normal(0, 0.05, 400),      # champion
             "near": y + rng.normal(0, 0.052, 400),     # indistinguishable from it
             "bad": y + rng.normal(0, 0.25, 400)}       # clearly worse
    for name, p in preds.items():
        np.savez(tmp_path / "predictions" / f"{name}.npz", holdout_pred=p, holdout_y=y,
                 oof=p, holdout_price=np.exp(y))
    pd.DataFrame({"model": ["good", "near", "bad"]}).to_csv(tmp_path / "results.csv", index=False)

    st = comparison_stats(tmp_path, n_boot=500, seed=1).set_index("model")
    assert st["tied_with_champion"].loc[["good", "near"]].all()
    assert not st.loc["bad", "tied_with_champion"]
    assert st.loc["bad", "delta_ci95_low"] > 0
    assert (st["mape_ci95_low"] <= st["holdout_mape_pct"]).all()
    assert (st["holdout_mape_pct"] <= st["mape_ci95_high"]).all()
    assert (tmp_path / "comparison_stats.csv").exists()


def test_storage_backend_detection(monkeypatch):
    from appraisenet import db
    monkeypatch.setenv("APPRAISENET_DB", "postgresql://user:secret@dbhost:5432/appraisenet")
    assert db.is_postgres()
    assert "secret" not in db.describe() and "dbhost" in db.describe()
    monkeypatch.setenv("APPRAISENET_DB", "data/listings.db")
    assert not db.is_postgres()


def test_daily_ingest_dedup(tmp_path, monkeypatch, cfg):
    from appraisenet import db
    monkeypatch.setenv("APPRAISENET_DB", str(tmp_path / "grow.db"))
    full = synthetic_listings(400, seed=11)
    full.iloc[:300].to_csv(tmp_path / "day1.csv", index=False)
    full.to_csv(tmp_path / "day2.csv", index=False)

    day1 = db.ingest(tmp_path / "day1.csv", cfg.protocol)
    assert 250 < day1["inserted"] <= day1["passed_gates"]
    again = db.ingest(tmp_path / "day1.csv", cfg.protocol)
    assert again["inserted"] == 0 and again["duplicates"] == day1["inserted"]
    day2 = db.ingest(tmp_path / "day2.csv", cfg.protocol)
    assert day2["inserted"] > 0
    assert day2["total_listings"] == day1["inserted"] + day2["inserted"]

    df, synthetic = load_listings()
    assert not synthetic and len(df) == day2["total_listings"]
    assert df["id"].is_unique


def test_registry_and_api(tmp_path, monkeypatch, cfg):
    # production fit + serve on synthetic, in an isolated models dir
    import appraisenet.registry as reg
    monkeypatch.setattr(reg, "CURRENT", tmp_path / "current")
    monkeypatch.setattr(reg, "ARCHIVE", tmp_path / "archive")
    c = load_config(None)
    c.protocol.folds = 2
    c.max_rows = 600
    meta = reg.train_production(c, force=True)
    assert meta["decision"] == "promoted" and meta["version"] == "1.0.0"

    from fastapi.testclient import TestClient

    import appraisenet.serve as srv
    monkeypatch.setattr(srv, "PRED_DB", tmp_path / "pred.db")
    srv._model = None
    client = TestClient(srv.app)
    r = client.post("/predict", json={"year": 2020, "make": "Toyoda", "model": "Camrio",
                                      "mileage": 60000, "seller_type": "dealer"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["low"] <= body["price"] <= body["high"]
    assert client.get("/health").json()["ok"] is True
