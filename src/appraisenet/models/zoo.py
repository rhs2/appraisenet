"""The model zoo: every learner behind one interface.

    fit_predict(name, train_df, eval_df, fs, y_col, cfg) -> predictions in log-price space

Families: linear (ridge / elastic net on compact one-hot), instance-based (k-NN
comparables), bagged trees (RandomForest / ExtraTrees), gradient boosting (LightGBM /
XGBoost / CatBoost, native categoricals), deep tabular (entity-embedding MLP and a
compact FT-Transformer), a text hybrid (champion + TF-IDF ridge on the residual of the
scrubbed description), a two-library blend, and a stacked ensemble whose meta-learner is
fitted on inner out-of-fold predictions only.

Hyper-parameters are sensible fixed defaults (documented per model): the study compares
model families under equal, honest conditions rather than per-family tuning budgets;
the champion's tuned production configuration is reported separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data import TARGET
from ..features import FeatureSpace


def _lgbm_params():
    return dict(objective="regression", num_leaves=63, learning_rate=0.03, min_data_in_leaf=20,
                feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
                verbosity=-1, seed=42, num_threads=-1)


def fit_predict(name: str, train: pd.DataFrame, ev: pd.DataFrame, fs: FeatureSpace,
                cfg: Config, y_col: str = TARGET) -> np.ndarray:
    y = train[y_col].values
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=3.0).fit(fs.onehot(train), y).predict(fs.onehot(ev))
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=5000).fit(fs.onehot(train), y).predict(fs.onehot(ev))
    if name == "knn_comparables":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=15, weights="distance").fit(fs.onehot(train), y).predict(fs.onehot(ev))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                  n_jobs=-1, random_state=42)
        return m.fit(fs.dense(train), y).predict(fs.dense(ev))
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        m = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                                n_jobs=-1, random_state=42)
        return m.fit(fs.dense(train), y).predict(fs.dense(ev))
    if name == "lightgbm":
        import lightgbm as lgb
        m = lgb.train(_lgbm_params(), lgb.Dataset(fs.tree_frame(train), y), num_boost_round=1500)
        return m.predict(fs.tree_frame(ev))
    if name == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8, subsample=0.9,
                         colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0,
                         tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                         n_jobs=-1, random_state=42, verbosity=0)
        return m.fit(fs.tree_frame(train), y).predict(fs.tree_frame(ev))
    if name == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=2500, learning_rate=0.05, depth=8, loss_function="RMSE",
                              random_seed=42, verbose=0, thread_count=-1, cat_features=fs.cat_cols)
        return m.fit(fs.catboost_frame(train), y).predict(fs.catboost_frame(ev))
    if name == "embed_mlp":
        from .nets import EmbedMLP
        ct, nt = fs.embed_tensors(train)
        ce, ne = fs.embed_tensors(ev)
        m = EmbedMLP(fs.cardinalities, nt.shape[1], epochs=cfg.torch_epochs, batch=cfg.torch_batch)
        return m.fit(ct, nt, y).predict(ce, ne)
    if name == "ft_transformer":
        from .nets import FTTransformer
        ct, nt = fs.embed_tensors(train)
        ce, ne = fs.embed_tensors(ev)
        m = FTTransformer(fs.cardinalities, nt.shape[1], epochs=cfg.torch_epochs, batch=cfg.torch_batch)
        return m.fit(ct, nt, y).predict(ce, ne)
    if name == "blend_lgbm_catboost":
        a = fit_predict("lightgbm", train, ev, fs, cfg, y_col)
        b = fit_predict("catboost", train, ev, fs, cfg, y_col)
        return (a + b) / 2.0
    if name == "hybrid_lgbm_text":
        return _hybrid_text(train, ev, fs, cfg, y_col)
    if name == "stack":
        return _stack(train, ev, fs, cfg, y_col)
    if name == "anchored_lgbm":
        return _anchored_lgbm(train, ev, fs, cfg, y_col)
    if name == "anchored_hybrid":
        return _anchored_hybrid(train, ev, fs, cfg, y_col)
    if name == "anchored_blend":
        return _anchored_blend(train, ev, fs, cfg, y_col)
    raise KeyError(f"unknown model '{name}'")


ZOO = ["ridge", "elasticnet", "knn_comparables", "random_forest", "extra_trees",
       "lightgbm", "xgboost", "catboost", "embed_mlp", "ft_transformer",
       "blend_lgbm_catboost", "hybrid_lgbm_text", "stack", "anchored_lgbm",
       "anchored_hybrid", "anchored_blend"]
FAMILY = {"ridge": "linear", "elasticnet": "linear", "knn_comparables": "instance",
          "random_forest": "bagged trees", "extra_trees": "bagged trees",
          "lightgbm": "boosting", "xgboost": "boosting", "catboost": "boosting",
          "embed_mlp": "deep tabular", "ft_transformer": "deep tabular",
          "blend_lgbm_catboost": "hybrid", "hybrid_lgbm_text": "hybrid", "stack": "hybrid",
          "anchored_lgbm": "anchored", "anchored_hybrid": "anchored",
          "anchored_blend": "anchored"}


def _hybrid_text(train, ev, fs, cfg, y_col):
    """LightGBM + a bounded TF-IDF ridge fitted on the tabular model's residual.

    The residual stage is fitted on inner out-of-fold residuals (never its own training
    predictions), and its correction is clipped to +-15% so marketing copy can only
    nudge a price, never invent one.
    """
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    y = train[y_col].values
    rng = np.random.RandomState(42)
    fold = rng.randint(0, 3, len(train))
    oof = np.zeros(len(train))
    for k in range(3):
        tr = train[fold != k]
        m = lgb.train(_lgbm_params(), lgb.Dataset(fs.tree_frame(tr), tr[y_col].values), num_boost_round=1200)
        oof[fold == k] = m.predict(fs.tree_frame(train[fold == k]))
    resid = y - oof
    vec = TfidfVectorizer(min_df=20, max_features=cfg.text_max_features, ngram_range=(1, 2), sublinear_tf=True)
    Xt = vec.fit_transform(train["description"].fillna("").str.lower())
    reg = Ridge(alpha=30.0).fit(Xt, resid)
    base = lgb.train(_lgbm_params(), lgb.Dataset(fs.tree_frame(train), y), num_boost_round=1500)
    correction = reg.predict(vec.transform(ev["description"].fillna("").str.lower()))
    return base.predict(fs.tree_frame(ev)) + np.clip(correction, -0.15, 0.15)


def _stack(train, ev, fs, cfg, y_col):
    """Ridge meta-learner over inner OOF predictions of the strongest base family mix."""
    from sklearn.linear_model import Ridge

    bases = ["lightgbm", "xgboost", "extra_trees"]   # catboost excluded: 5x the fit cost for
    rng = np.random.RandomState(42)                  # no blend gain beyond the lgbm pair
    fold = rng.randint(0, 3, len(train))
    oof = np.zeros((len(train), len(bases)))
    for j, b in enumerate(bases):
        for k in range(3):
            tr, ho = train[fold != k], train[fold == k]
            oof[fold == k, j] = fit_predict(b, tr, ho, fs, cfg, y_col)
    meta = Ridge(alpha=1.0).fit(oof, train[y_col].values)
    evp = np.column_stack([fit_predict(b, train, ev, fs, cfg, y_col) for b in bases])
    return meta.predict(evp)


class _AnchorLadder:
    """Fold-fitted group-median anchors, the way deployed pricing systems seed a learner.

    A ladder of log-price medians computed on the TRAINING rows only: trim group ->
    model + 2-year window -> model line -> make + body style -> global. Each row gets the
    first anchor its group supports, plus a train-fitted state price index. Fitted per
    fold like every other statistic in the protocol, so it can never leak evaluation rows.
    """

    LADDER = [(["make", "model", "trim"], 8), (["make", "model", "_yb"], 8),
              (["make", "model"], 5), (["make", "body_style"], 30)]

    @staticmethod
    def _key(df, cols):
        s = df[cols[0]].astype("string").str.lower().fillna("~")
        for c in cols[1:]:
            s = s + "|" + df[c].astype("string").str.lower().fillna("~")
        return s

    @staticmethod
    def _prep(df):
        d = df.copy()
        d["_yb"] = (pd.to_numeric(d["year"], errors="coerce") // 2).astype("Int64").astype("string")
        return d

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "_AnchorLadder":
        d = self._prep(train)
        lp = pd.Series(y, index=d.index)
        self.maps = []
        for cols, min_n in self.LADDER:
            g = lp.groupby(self._key(d, cols)).agg(["median", "count"])
            self.maps.append((cols, g.loc[g["count"] >= min_n, "median"]))
        self.global_med = float(np.median(y))
        st = lp.groupby(d["region_state"].astype("string").str.upper().fillna("~")).agg(["median", "count"])
        self.state_idx = (st.loc[st["count"] >= 300, "median"] - self.global_med)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._prep(df)
        anchor = pd.Series(np.nan, index=d.index)
        for cols, m in self.maps:
            need = anchor.isna()
            if not need.any():
                break
            anchor[need] = self._key(d[need], cols).map(m)
        anchor = anchor.fillna(self.global_med)
        sidx = d["region_state"].astype("string").str.upper().fillna("~").map(self.state_idx).fillna(0.0)
        return pd.DataFrame({"log_anchor": anchor.values, "state_idx": sidx.values}, index=df.index)


def fit_anchor_ladder(train: pd.DataFrame, y: np.ndarray) -> "_AnchorLadder":
    """Fit the anchor ladder on training rows. Public because the production registry
    persists a ladder alongside the booster and applies it at serving time."""
    return _AnchorLadder().fit(train, y)


def anchored_frame(fs: FeatureSpace, df: pd.DataFrame, ladder: "_AnchorLadder") -> pd.DataFrame:
    """The tree frame plus the two anchor columns, in the order the booster expects."""
    return pd.concat([fs.tree_frame(df), ladder.transform(df)], axis=1)


def anchored_params(columns) -> dict:
    """LightGBM parameters for the anchored configuration: the study defaults plus
    monotone decreasing constraints on mileage and miles per year."""
    p = _lgbm_params()
    p["monotone_constraints"] = [(-1 if c in ("mileage", "miles_per_year") else 0) for c in columns]
    return p


def _anchored_lgbm(train, ev, fs, cfg, y_col):
    """LightGBM seeded with fold-fitted anchors + a monotone mileage constraint.

    Mirrors the configuration deployed pricing systems actually run: the anchor ladder
    hands the booster a market baseline for every car (so rare models are priced against
    their group, not extrapolated), and monotone constraints on mileage and miles_per_year
    encode that more miles can never raise a price, all else equal. This is the
    configuration `train-production` deploys, fitted here through the same code path.
    """
    import lightgbm as lgb

    y = train[y_col].values
    ladder = fit_anchor_ladder(train, y)
    Xt, Xe = anchored_frame(fs, train, ladder), anchored_frame(fs, ev, ladder)
    m = lgb.train(anchored_params(Xt.columns), lgb.Dataset(Xt, y), num_boost_round=1500)
    return m.predict(Xe)


class _Zip3TE:
    """Smoothed target encoding of the 3-digit region, fitted on training rows only.

    te(z) = (sum_z + m * global_mean) / (n_z + m), m = 50: a region with few cars shrinks
    to the global mean, a region with thousands speaks for itself. For TRAINING rows the
    encoding is built inner-out-of-fold (5 inner folds; each row is encoded by the other
    folds), the standard guard against a target encoding memorising its own labels.
    """

    def __init__(self, m: float = 50.0):
        self.m = m

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "_Zip3TE":
        z = df["region_zip3"].astype("string").fillna("~")
        g = pd.DataFrame({"z": z.values, "y": y}).groupby("z")["y"].agg(["sum", "count"])
        self.global_mean = float(np.mean(y))
        self.sums, self.counts = g["sum"], g["count"]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = df["region_zip3"].astype("string").fillna("~")
        s = z.map(self.sums).fillna(0.0).astype(float)
        c = z.map(self.counts).fillna(0.0).astype(float)
        return ((s + self.m * self.global_mean) / (c + self.m)).values


def _anchor_setup(train, ev, fs, y_col):
    """Shared stage 1 for the residual-to-anchor engines: fold-fitted ladder, residual
    target, inner-OOF zip3 target encoding, and the augmented tree frames."""
    y = train[y_col].values
    ladder = _AnchorLadder().fit(train, y)
    at, ae = ladder.transform(train), ladder.transform(ev)
    r = y - at["log_anchor"].values

    rng = np.random.RandomState(42)
    inner = rng.randint(0, 5, len(train))
    te_tr = np.zeros(len(train))
    for k in range(5):
        m = inner == k
        te_tr[m] = _Zip3TE().fit(train[~m], r[~m]).transform(train[m])
    te_full = _Zip3TE().fit(train, r)

    def frame(df, anchors, te_vals):
        X = fs.tree_frame(df).copy()
        X["log_anchor"] = anchors["log_anchor"].values
        X["state_idx"] = anchors["state_idx"].values
        X["zip3_te"] = te_vals
        return X

    return r, ae, frame(train, at, te_tr), frame(ev, ae, te_full.transform(ev))


def _resid_lgbm(Xt, r, Xe):
    """The monotone LightGBM residual engine over anchor-augmented frames."""
    import lightgbm as lgb
    p = _lgbm_params()
    p["monotone_constraints"] = [(-1 if c in ("mileage", "miles_per_year") else 0) for c in Xt.columns]
    m = lgb.train(p, lgb.Dataset(Xt, r), num_boost_round=1500)
    return m.predict(Xe)


def _anchored_hybrid(train, ev, fs, cfg, y_col):
    """The robust production hybrid: the anchor as an OFFSET, not just a feature.

    The deployed configuration this mirrors prices a car as its market group's anchor
    plus a learned, bounded deviation. Stage 1 is the fold-fitted anchor ladder; the
    booster (same LightGBM configuration, monotone mileage) is then trained on the
    RESIDUAL y - log_anchor, with the anchor features and an inner-OOF zip3 target
    encoding of the regional residual among its inputs. The final prediction is
    anchor + clip(residual_hat, +-0.75): a car can never be priced further than about
    2.1x away from its own market group, which is what keeps rare cars robust.
    """
    r, ae, Xt, Xe = _anchor_setup(train, ev, fs, y_col)
    return ae["log_anchor"].values + np.clip(_resid_lgbm(Xt, r, Xe), -0.75, 0.75)


def _anchored_blend(train, ev, fs, cfg, y_col):
    """Two residual-to-anchor engines blended: the bakeoff configuration deployed
    systems converge on.

    The same stage 1 as the anchored hybrid feeds two independent residual engines:
    the monotone LightGBM, and CatBoost (the study's standard CatBoost configuration)
    on raw category strings. Their residual predictions are averaged in log space (a
    geometric mean in price space) before the same +-0.75 bound is applied. Two engines
    with different categorical handling make uncorrelated mistakes on thin groups;
    the average keeps the strengths of both.
    """
    from catboost import CatBoostRegressor

    r, ae, Xt, Xe = _anchor_setup(train, ev, fs, y_col)
    r_lgb = _resid_lgbm(Xt, r, Xe)

    def cb_frame(X):
        C = X.copy()
        for c in fs.cat_cols:
            C[c] = C[c].astype("string").fillna("NA").astype(str)
        return C
    cb = CatBoostRegressor(iterations=2500, learning_rate=0.05, depth=8, loss_function="RMSE",
                           random_seed=42, verbose=0, thread_count=-1, cat_features=fs.cat_cols)
    cb.fit(cb_frame(Xt), r)
    r_cat = cb.predict(cb_frame(Xe))
    return ae["log_anchor"].values + np.clip(0.5 * r_lgb + 0.5 * r_cat, -0.75, 0.75)


def quantile_models(train: pd.DataFrame, fs: FeatureSpace, y_col: str = TARGET):
    """LightGBM p10/p90 for the production interval (conformalised in protocol.py)."""
    import lightgbm as lgb
    out = {}
    for alpha, tag in [(0.1, "p10"), (0.9, "p90")]:
        p = _lgbm_params()
        p.update(objective="quantile", alpha=alpha)
        out[tag] = lgb.train(p, lgb.Dataset(fs.tree_frame(train), train[y_col].values), num_boost_round=1200)
    return out
