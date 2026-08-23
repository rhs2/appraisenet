"""Feature preparation, fitted on training rows only.

Three encoders serve three model families from one source of truth:
  - `TreeFrame`: pandas categoricals with frozen levels (LightGBM / XGBoost / CatBoost
    consume categoricals natively; CatBoost receives the raw strings).
  - `DenseMatrix`: ordinal-coded categoricals + median-imputed numerics for sklearn
    estimators, with an optional one-hot variant for linear models.
  - `EmbedTensors`: integer codes (0 = unknown/missing) + standardised numerics for the
    PyTorch models.

`TrimTier` is the one engineered categorical: the trim's price positioning within its
model line (top / upper / mid / base), computed from training-fold group medians only,
so it is a fitted transform and can never leak the evaluation rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CATEGORICAL, NUMERIC


class TrimTier:
    """Within-model price positioning of a trim, fitted on train only."""

    def __init__(self, min_group: int = 4):
        self.min_group = min_group
        self.tiers: dict[str, str] = {}

    @staticmethod
    def _key(model, trim) -> str | None:
        if pd.isna(model) or pd.isna(trim):
            return None
        return f"{str(model).lower()}|{str(trim).lower()}"

    def fit(self, df: pd.DataFrame) -> TrimTier:
        d = df.dropna(subset=["model", "trim"]).copy()
        if d.empty:
            return self
        d["lp"] = np.log(d["price"])
        model_med = d.groupby(d["model"].str.lower())["lp"].transform("median")
        d["rel"] = d["lp"] - model_med
        g = d.groupby([d["model"].str.lower(), d["trim"].str.lower()])["rel"].agg(["median", "count"])
        for (m, t), r in g.iterrows():
            if r["count"] < self.min_group:
                continue
            rel = r["median"]
            self.tiers[f"{m}|{t}"] = "top" if rel >= 0.10 else "upper" if rel >= 0.03 else "base" if rel <= -0.03 else "mid"
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        keys = [self._key(m, t) for m, t in zip(df["model"], df["trim"])]
        return pd.Series([self.tiers.get(k, "unknown") if k else "unknown" for k in keys],
                         index=df.index, name="trim_tier")


@dataclass
class FeatureSpace:
    """Fitted on the training rows; every view mirrors the same columns and levels."""

    cat_cols: list[str] = field(default_factory=lambda: CATEGORICAL + ["trim_tier"])
    num_cols: list[str] = field(default_factory=lambda: list(NUMERIC))
    levels: dict[str, list[str]] = field(default_factory=dict)
    num_median: pd.Series | None = None
    num_mean: pd.Series | None = None
    num_std: pd.Series | None = None
    trim_tier: TrimTier = field(default_factory=TrimTier)

    def fit(self, train: pd.DataFrame) -> FeatureSpace:
        self.trim_tier.fit(train)
        t = self._with_tier(train)
        for c in self.cat_cols:
            self.levels[c] = sorted(t[c].dropna().astype(str).unique().tolist())
        num = t[self.num_cols].apply(pd.to_numeric, errors="coerce")
        self.num_median = num.median()
        self.num_mean = num.mean()
        self.num_std = num.std().replace(0, 1.0).fillna(1.0)
        return self

    def _with_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["trim_tier"] = self.trim_tier.transform(df)
        return out

    # ---- views ----
    def tree_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        t = self._with_tier(df)
        X = t[self.num_cols + self.cat_cols].copy()
        for c in self.cat_cols:
            X[c] = pd.Categorical(X[c].astype("string"), categories=self.levels[c])
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def catboost_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        t = self._with_tier(df)
        X = t[self.num_cols + self.cat_cols].copy()
        for c in self.cat_cols:
            X[c] = X[c].astype("string").fillna("NA").astype(str)
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def dense(self, df: pd.DataFrame) -> np.ndarray:
        t = self._with_tier(df)
        num = t[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.num_median).fillna(0.0)
        cats = []
        for c in self.cat_cols:
            idx = {v: i + 1 for i, v in enumerate(self.levels[c])}
            cats.append(t[c].astype("string").map(idx).fillna(0).astype(float))
        return np.column_stack([num.values] + [s.values for s in cats])

    def onehot(self, df: pd.DataFrame, top_k: int = 40) -> np.ndarray:
        """Compact one-hot for linear models: top-K levels per categorical."""
        t = self._with_tier(df)
        num = t[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.num_median)
        num = ((num - self.num_mean) / self.num_std).fillna(0.0)
        blocks = [num.values]
        for c in self.cat_cols:
            keep = self.levels[c][:top_k]
            s = t[c].astype("string")
            blocks.append(np.column_stack([(s == v).fillna(False).astype(float).values for v in keep]) if keep
                          else np.zeros((len(t), 0)))
        return np.column_stack(blocks)

    def embed_tensors(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(cat_codes int64 [n, C], numeric standardised float32 [n, N]); 0 = unknown."""
        t = self._with_tier(df)
        codes = []
        for c in self.cat_cols:
            idx = {v: i + 1 for i, v in enumerate(self.levels[c])}
            codes.append(t[c].astype("string").map(idx).fillna(0).astype(np.int64).values)
        num = t[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.num_median)
        num = ((num - self.num_mean) / self.num_std).fillna(0.0).astype(np.float32)
        return np.column_stack(codes), num.values

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.levels[c]) + 1 for c in self.cat_cols]
