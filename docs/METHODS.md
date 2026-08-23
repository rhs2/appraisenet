# Methods

The complete experimental methodology, at the level of detail a paper's Methods section
requires. Everything here is implemented in `src/appraisenet/` and can be re-executed
with one command; nothing is described that the code does not do.

## 1. Problem formulation

Given a listing `x` (vehicle identity, specifications, mileage, seller type, coarse
location, free-text description), predict its asking price. All models are trained on
`y = log(price)`. Price errors are relative by nature: a $500 miss is a serious error
on a cheap car yet negligible on an expensive one, so modeling in log space makes
errors mean the same thing across the study's whole $2,000-$100,000 price band, and
`exp()` of a symmetric interval in log space yields the asymmetric dollar interval a
buyer actually needs.

Metrics, computed in price space from log-space predictions p and targets y:

- absolute percentage error per car: `APE_i = |exp(p_i - y_i) - 1|`
- MAPE (mean), median APE, share of cars within 10%
- R2 in log space
- interval coverage (share of holdout cars inside their interval) and median interval
  width as a percentage of the true price

## 2. Data

38,758 US used-vehicle listings (July-August 2026, dealer and private-party sellers),
asking prices in $2,000-$100,000, model year 1990+. The corpus is proprietary and not
distributed; identity was stripped before it reached this project. Curation reduced
roughly 160 raw fields per listing to 26 modeling columns through field triage,
VIN-decode enrichment (NHTSA vPIC), junk-price and not-a-car removal, per-vehicle
deduplication and a label-noise quarantine; `data/README.md` documents every step.
A synthetic generator with the identical schema and plausible price physics stands in
for the corpus in tests and CI.

Engineered per row before splitting: `age = current_year - year` (clipped at 0) and
`miles_per_year = mileage / max(age, 1)`. The target `log_price` is set after the
price-band filter, so the band never peeks at evaluation labels.

## 3. Evaluation protocol

- **Split**: a 10% holdout drawn once with seed 42. It is scored exactly once per model
  and never used for any selection decision.
- **Cross-validation**: the remaining 90% is assigned to 5 folds (seed 42). Out-of-fold
  (OOF) predictions supply the selection metrics and the conformal calibration
  residuals.
- **Per-fold fitting of every fitted statistic**: the `FeatureSpace` (categorical
  vocabularies, imputation medians, standardisation moments) and the engineered
  `trim_tier` feature are re-fitted on each fold's training rows. `trim_tier` ranks a
  trim's median log-price within its model line (top / upper / mid / base, groups with
  fewer than 4 training examples fall back to "unknown"); computed on all data it would
  leak the target, so it is treated as a model parameter, never a data column.
- **Final fit**: after cross-validation, each model is refitted on the full training
  partition and predicts the holdout. Only those numbers are quoted as performance.
- **No per-model tuning**: every model runs with fixed, documented hyper-parameters
  (Section 5). The study compares model families under equal conditions rather than
  tuning budgets; the champion's production configuration is reported separately.

## 4. Feature spaces

One fitted `FeatureSpace` object serves four views, so every family sees the same
information in its natural encoding:

| view | encoding | consumers |
|---|---|---|
| `tree_frame` | pandas categoricals with frozen training levels; unseen levels become missing, never new categories | LightGBM, XGBoost |
| `catboost_frame` | raw category strings ("NA" for missing) | CatBoost |
| `dense` | ordinal codes (0 = unseen) + median-imputed numerics | RandomForest, ExtraTrees |
| `onehot` | top-40 levels per categorical + z-scored numerics | ridge, elastic net, k-NN |
| `embed_tensors` | integer codes (0 = unknown) + z-scored numerics | EmbedMLP, FT-Transformer |

Categoricals: make, model, trim, body_style, drivetrain, transmission, fuel_type,
electrification, gvwr_class, series, plant_country, adaptive_cruise, seller_type,
region_state, trim_tier. Numerics: mileage, age, miles_per_year, doors, cylinders,
engine_hp, displacement_l, original_price.

## 5. Models and exact hyper-parameters

| model | configuration |
|---|---|
| ridge | alpha 3.0, one-hot view |
| elastic net | alpha 0.001, l1_ratio 0.3, max_iter 5000, one-hot view |
| k-NN comparables | k = 15, distance weighting, one-hot view |
| RandomForest | 300 trees, min_samples_leaf 3, max_features 0.5 |
| ExtraTrees | 400 trees, min_samples_leaf 2, max_features 0.5 |
| LightGBM | 1,500 rounds, lr 0.03, 63 leaves, min_data_in_leaf 20, feature_fraction 0.8, bagging 0.9 (freq 1), lambda_l2 1.0 |
| XGBoost | 2,000 trees, lr 0.03, depth 8, subsample 0.9, colsample_bytree 0.7, min_child_weight 5, reg_lambda 1.0, hist, native categoricals |
| CatBoost | 2,500 iterations, lr 0.05, depth 8, RMSE, native categorical handling |
| EmbedMLP | per-categorical embeddings of dim min(32, max(4, round(1.6 c^0.56))); trunk BatchNorm -> 256 GELU (dropout 0.15) -> 128 GELU (dropout 0.10) -> 1 |
| FT-Transformer | token dim 48, 3 pre-norm encoder layers, 6 heads, FFN 4x, dropout 0.1; numerics linearly tokenised; [CLS] pooling |
| blend | arithmetic mean of LightGBM and CatBoost predictions |
| text hybrid | Section 5.2 |
| stack | Section 5.3 |

### 5.1 Deep models: training recipe

Both networks train with AdamW (lr 1e-3, weight decay 1e-4), cosine-annealed over at
most 40 epochs, batch 1024, Huber loss (delta 1.0), early stopping with patience 6 on
a 10% validation slice of the training fold, best-epoch weights restored. The target is
standardised to z-space before training and un-standardised at prediction; log-price
sits near 9-10, far from a randomly initialised network's output scale, and training in
z-space converges in a handful of epochs where raw-target training needs dozens. Runs
on Apple MPS or CUDA when available. Both architectures are implemented in-repo in
PyTorch (about 150 lines) rather than through wrapper libraries.

### 5.2 Text hybrid

A bounded residual corrector on the scrubbed description text. LightGBM (1,200 rounds)
produces 3-fold inner-OOF predictions on the training partition; a TF-IDF ridge
(1-2 grams, min_df 20, max 20,000 features, sublinear TF, alpha 30) is fitted on the
inner-OOF residuals, never on residuals of predictions that saw the same rows. At
inference the correction is clipped to +-0.15 in log space (about +-14% in price):
listing text may nudge a price, never invent one.

### 5.3 Stacked ensemble

Bases: LightGBM, XGBoost, ExtraTrees (CatBoost excluded: about 5x the fit cost for no
gain beyond the blend pair). Each base produces 3-fold inner-OOF predictions on the
training partition; a ridge meta-learner (alpha 1.0) is fitted on those inner-OOF
columns only. For evaluation, the bases are refitted on the full training partition and
the meta-learner combines their predictions. The meta-learner never trains on
predictions a base made for rows it was fitted on.

## 6. Uncertainty quantification

**Split-conformal intervals (every model).** With OOF absolute residuals
`r_1..r_n` in log space, the widening constant is
`q = Quantile(r, ceil((n+1) * 0.8) / n)`, and the interval for a holdout car is
`[exp(p - q), exp(p + q)]`. Marginal 80% coverage is guaranteed distribution-free
under exchangeability; the holdout coverage table empirically verifies it.

**Conformalised quantile regression (production champion).** LightGBM quantile models
(alpha 0.1 and 0.9, 1,200 rounds) are trained per fold; conformity scores
`s_i = max(lo_i - y_i, y_i - hi_i)` on OOF predictions give a widening constant at the
same corrected quantile, added to both quantile predictions. CQR keeps the coverage
guarantee while letting the width adapt per car: wide for a rare old truck, narrow for
a common commuter sedan.

## 7. Statistical comparison

Every model predicts the same holdout cars, so model differences are judged on paired
errors. A paired bootstrap (4,000 resamples of holdout indices, the same draw applied
to every model, seed 42) yields a 95% percentile interval for each model's MAPE and for
its MAPE gap to the champion. A model whose gap interval includes zero is reported as
statistically tied with the champion; the study never claims a ranking the data cannot
support. Implemented in `src/appraisenet/compare.py`, re-runnable from the persisted
prediction arrays without refitting.

## 8. Production learning loop

- **Ingest** (`appraisenet data ingest`): new listings pass the same quality gates as
  training, are fingerprinted on identifying fields, and only never-seen rows are
  appended (idempotent; audited in `ingest_log`). Storage is SQLite or PostgreSQL
  behind one URL.
- **Registry** (`appraisenet train-production`): fits the champion configuration plus
  CQR, then promotes only if holdout MAPE is within 0.30 points of the serving model;
  otherwise the candidate is archived and the old version keeps serving. Semantic
  versions, automatic minor bumps, previous versions kept for rollback.
- **Serving** (`appraisenet serve`): FastAPI, point price + calibrated interval,
  hot-reload on promotion, every prediction logged.
- **Drift** (`appraisenet monitor`): population-stability index between the training
  reference and recent prediction traffic, numeric (decile bins) and categorical
  (top-20 levels), warn at 0.10 and alert at 0.25.

## 9. Reproducibility

- Seeds are fixed (42 throughout; deep models additionally seed torch).
- `appraisenet benchmark --config configs/default.yaml` re-runs the full study;
  `configs/smoke.yaml` is a 4-model subset for CI.
- Every artifact needed to rebuild tables and figures is persisted:
  `results.csv`, `comparison_stats.csv`, `segments.csv`, per-model prediction arrays,
  `run_meta.json`, model cards. `appraisenet report` rebuilds figures and statistics
  from artifacts alone.
- Without the private corpus, every command runs on the synthetic generator and is
  labelled as such in every output. CI never sees real data.

## References

- Guo, C. and Berkhahn, F. (2016). Entity Embeddings of Categorical Variables.
- Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A. (2021). Revisiting Deep
  Learning Models for Tabular Data (FT-Transformer).
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Prokhorenkova, L. et al. (2018). CatBoost: Unbiased Boosting with Categorical
  Features.
- Romano, Y., Patterson, E. and Candes, E. (2019). Conformalized Quantile Regression.
- Lei, J. et al. (2018). Distribution-Free Predictive Inference for Regression.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.
