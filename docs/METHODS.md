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

1,174,659 US used-vehicle listings (collected July-August 2026, dealer and
private-party sellers, growing daily through the ingest path), asking prices in
$2,000-$100,000, model year 1990+. The corpus is proprietary and not distributed;
identity was stripped before it reached this project. Curation reduced roughly 160 raw
fields per listing to 29 modeling columns through field triage, VIN-decode enrichment
(NHTSA vPIC), junk-price and not-a-car removal, per-vehicle deduplication and a
label-noise quarantine; `data/README.md` documents every step, including the deliberate
exclusion of the listing's first asking price and derived price-drop fields, which
would let a model read the label rather than price the car. A synthetic generator with
the identical schema and plausible price physics stands in for the corpus in tests and
CI.

The study's first edition (the **pilot**) ran the entire 13-model zoo on the
38,758-listing corpus snapshot of late August 2026; its committed evidence lives in
`reports/pilot/`. The corpus-scale edition runs every model that can ride the full
corpus on a single machine (`configs/default.yaml`); the classical baselines whose
memory or distance computations grow quadratically at this scale (linear one-hot,
k-NN, unbounded-depth bagged trees) remain represented by their pilot-tier results
(`configs/pilot.yaml` reproduces that tier).

Engineered per row before splitting: `age = current_year - year` (clipped at 0) and
`miles_per_year = mileage / max(age, 1)`. The target `log_price` is set after the
price-band filter, so the band never peeks at evaluation labels.

## 3. The AppraiseNet Evaluation Protocol (AEP)

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
engine_hp, displacement_l, original_price, msrp, days_listed, price_changes.

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
| anchored LightGBM | Section 5.4 |
| anchored hybrid | Section 5.5 |
| anchored blend | Section 5.6 |

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

### 5.4 Anchored LightGBM

The configuration deployed pricing systems actually run, brought into the study under
the same protocol. A ladder of log-price group medians is fitted on the training rows
only: trim group (make, model, trim; at least 8 rows) -> model plus 2-year window (at
least 8) -> model line (at least 5) -> make plus body style (at least 30) -> global
median. Every car receives the first anchor its group supports (`log_anchor`), plus a
state price index (state median minus global, states with at least 300 training rows).
The booster is the study's standard LightGBM configuration with two additions: the two
anchor features, and monotone decreasing constraints on `mileage` and `miles_per_year`
(more miles can never raise a price, all else equal). Like `trim_tier`, the ladder is a
fitted transform, re-learned inside every fold, so it can never leak evaluation rows.
The anchor arrives as information only: the learner predicts the price directly and may
disagree with the anchor by any amount. This is the configuration `train-production`
deploys, fitted through the same code path (`zoo.fit_anchor_ladder`, `zoo.anchored_frame`,
`zoo.anchored_params`) in both places.

### 5.5 Anchored hybrid

The robust configuration deployed pricing systems converge on, with the anchor as an
**offset** rather than a feature. Stage 1 is the same fold-fitted anchor ladder as
Section 5.4. The booster (the study's standard LightGBM with the same monotone
constraints) is then trained on the residual `y - log_anchor`, seeing the anchor
features plus a smoothed target encoding of the 3-digit region:
`te(z) = (sum_z + 50 * global_mean) / (n_z + 50)`, computed on the regional residuals so
it captures location effects beyond what the anchor already explains. For training rows
the encoding is built inner-out-of-fold (5 inner folds, each row encoded by the other
four), the standard guard against a target encoding memorising its own labels; for
evaluation rows it is fitted on the full training partition. The final prediction is
`log_anchor + clip(residual_hat, -0.75, +0.75)`: a car can never be priced further than
about 2.1x from its own market group's anchor, which is what keeps rare and
thinly-supported cars robust instead of extrapolated. The bound is a fixed constant,
applied identically to a common sedan and to a fifteen-year-old truck. At corpus scale
that costs about a MAPE point, entirely below $10,000, and the study reports the
measurement rather than repairing the configuration after seeing the holdout: the
ladder's first rung groups make, model and trim without a model-year term, so an old car
inherits an anchor set by its newer siblings and the bound then forbids the learner from
descending to its real price. 96% of this design's misses beyond 50% are
over-predictions. A year-aware rung, or a bound scaled by within-group price dispersion,
is the obvious correction and is future work rather than a silent edit to these numbers.

### 5.6 Anchored blend

Two independent residual-to-anchor engines over the identical stage 1 of Section 5.5:
the monotone LightGBM, and the study's standard CatBoost configuration on raw category
strings. Their residual predictions are averaged in log space (a geometric mean in
price space) before the same +-0.75 bound is applied. The rationale is engine
diversity: two boosters with different categorical handling make uncorrelated mistakes
on thin groups, and averaging keeps the strengths of both. This is the configuration an
internal engine bakeoff of five candidates (log-price, log-price + monotone, residual +
monotone, residual blend, comps-blended) selected for deployment-style robustness;
comps blending, the fifth candidate, degraded accuracy and survives only as the anchor
ladder itself.

At corpus scale the deep models train with batch 4096 for at most 20 epochs (the same
early stopping applies); at 30x the pilot's data volume an epoch sees vastly more
examples, and the pilot's 40-epoch budget would be wasted compute.

## 6. Uncertainty quantification

**Split-conformal intervals (every model).** With OOF absolute residuals
`r_1..r_n` in log space, the widening constant is
`q = Quantile(r, ceil((n+1) * 0.8) / n)`, and the interval for a holdout car is
`[exp(p - q), exp(p + q)]`. Marginal 80% coverage is guaranteed distribution-free
under exchangeability; the holdout coverage table empirically verifies it.

**Conformalised quantile regression (production configuration).** LightGBM quantile models
(alpha 0.1 and 0.9, 1,200 rounds) are trained per fold; conformity scores
`s_i = max(lo_i - y_i, y_i - hi_i)` on OOF predictions give a widening constant at the
same corrected quantile, added to both quantile predictions. CQR keeps the coverage
guarantee while letting the width adapt per car: wide for a rare old truck, narrow for
a common commuter sedan.

## 7. Statistical comparison and error profiling

Every model predicts the same holdout cars, so model differences are judged on paired
errors. A paired bootstrap (4,000 resamples of holdout indices, the same draw applied
to every model, seed 42) yields a 95% percentile interval for each model's error and
for its gap to the leader. A model whose gap interval includes zero is reported as
statistically tied; the study never claims a ranking the data cannot support. Three
things are inferred from the same draws:

- **MAPE**, and each model's MAPE gap to the champion (`comparison_stats.csv`).
- **Median APE**, and each model's median gap to the best median (same file). MAPE is a
  mean and therefore a statement about the tail; the median is a statement about the
  typical car. At corpus scale the two summaries name different winners, so reporting
  only one of them would hide a real property of the field.
- **Every pair**, not only the gap to the leader (`pairwise_mape.csv`): with 10 models
  that is 45 intervals, because a claim about two models in the middle of the table
  needs its own interval.

Draws are generated one at a time rather than as one index matrix: 4,000 draws over a
117,260-car holdout would otherwise allocate several gigabytes.

`error_profiles` writes two further artifacts from the same prediction arrays, no
refitting involved:

- `error_profile.csv`: per model, the error percentiles (p50, p75, p90, p95, p99), the
  share of cars inside 5 / 10 / 20%, the share missed by more than 25 / 50 / 100%, the
  direction of the large misses (what fraction of misses beyond 50% are
  over-predictions), and the share of total absolute percentage error contributed by
  cars missed by more than 25%.
- `price_bands.csv`: mean and median APE per model in four half-open price bands. The
  bands are defined once in `compare.BANDS` and shared with the segment table, so a car
  priced at exactly $20,000 lands in exactly one of them.

All of it is implemented in `src/appraisenet/compare.py` and re-runnable from the
persisted prediction arrays with `appraisenet report`.

## 8. Production learning loop

- **Ingest** (`appraisenet data ingest`): new listings pass the same quality gates as
  training, are fingerprinted on identifying fields, and only never-seen rows are
  appended (idempotent; audited in `ingest_log`). Storage is SQLite or PostgreSQL
  behind one URL.
- **Registry** (`appraisenet train-production`): fits the production configuration plus
  CQR, then promotes only if holdout MAPE is within 0.30 points of the serving model;
  otherwise the candidate is archived and the old version keeps serving. Semantic
  versions, automatic minor bumps, previous versions kept for rollback. The production
  configuration is the best text-free model rather than the study's overall champion:
  the prediction API receives structured fields without a listing description, so the
  deployed model is selected for the inputs it will actually see. Since the corpus-scale
  study that model is the **anchored LightGBM** of Section 5.4 (`registry.CONFIGURATION`):
  statistically tied with the text hybrid it replaces, 0.025 MAPE points ahead of the
  plain booster it replaced in the previous edition, monotone in mileage, and 29x cheaper
  to fit than the champion blend. The staged artifact therefore contains `point.txt`,
  `p10.txt`, `p90.txt`, `feature_space.joblib` and `anchor_ladder.joblib`; the serving
  path applies the ladder to every request before predicting, and the quantile models
  stay on the plain frame, bounding the anchored point estimate rather than re-deriving
  it. A model staged before this change carries no ladder and is served unchanged.
- **Serving** (`appraisenet serve`): FastAPI, point price + calibrated interval,
  hot-reload on promotion, every prediction logged.
- **Drift** (`appraisenet monitor`): population-stability index between the training
  reference and recent prediction traffic, numeric (decile bins) and categorical
  (top-20 levels), warn at 0.10 and alert at 0.25.

## 9. Reproducibility

- Seeds are fixed (42 throughout; deep models additionally seed torch).
- `appraisenet benchmark --config configs/default.yaml` re-runs the full study;
  `configs/smoke.yaml` is a 4-model subset for CI.
- Every artifact needed to rebuild tables and figures is persisted: `results.csv`,
  `comparison_stats.csv`, `pairwise_mape.csv`, `error_profile.csv`, `price_bands.csv`,
  `segments.csv`, per-model prediction arrays, `run_meta.json`, model cards.
  `appraisenet report` rebuilds every statistic and figure from those artifacts alone,
  with no refitting.
- A corpus-scale run is checkpointed: `results.csv` and `predictions/*.npz` are written
  after every model, and a relaunch resumes from them. A finished model is reused only
  when its saved holdout provably matches the current split, so a changed corpus or
  protocol always refits. The corpus tier took 39 hours on one workstation.
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
