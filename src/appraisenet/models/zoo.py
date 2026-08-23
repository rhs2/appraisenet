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
    raise KeyError(f"unknown model '{name}'")


ZOO = ["ridge", "elasticnet", "knn_comparables", "random_forest", "extra_trees",
       "lightgbm", "xgboost", "catboost", "embed_mlp", "ft_transformer",
       "blend_lgbm_catboost", "hybrid_lgbm_text", "stack"]
FAMILY = {"ridge": "linear", "elasticnet": "linear", "knn_comparables": "instance",
          "random_forest": "bagged trees", "extra_trees": "bagged trees",
          "lightgbm": "boosting", "xgboost": "boosting", "catboost": "boosting",
          "embed_mlp": "deep tabular", "ft_transformer": "deep tabular",
          "blend_lgbm_catboost": "hybrid", "hybrid_lgbm_text": "hybrid", "stack": "hybrid"}


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


def quantile_models(train: pd.DataFrame, fs: FeatureSpace, y_col: str = TARGET):
    """LightGBM p10/p90 for the production interval (conformalised in protocol.py)."""
    import lightgbm as lgb
    out = {}
    for alpha, tag in [(0.1, "p10"), (0.9, "p90")]:
        p = _lgbm_params()
        p.update(objective="quantile", alpha=alpha)
        out[tag] = lgb.train(p, lgb.Dataset(fs.tree_frame(train), train[y_col].values), num_boost_round=1200)
    return out
