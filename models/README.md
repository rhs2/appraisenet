# Models

This directory is the production model registry. It ships empty on purpose: trained
weights are fitted on the proprietary corpus, so they are an artifact of the private
data and are never distributed with the code.

`appraisenet train-production` populates it:

```
models/
  current/            the serving version
    point.txt         LightGBM point-price model
    p10.txt, p90.txt  LightGBM quantile models behind the CQR interval
    feature_space.joblib  the fitted FeatureSpace (levels, imputation, trim tiers)
    meta.json         version, holdout metrics, conformal widening, decision
  archive/<stamp>/    every previously promoted version, kept for rollback
```

- **Promote-or-rollback**: a fresh candidate is promoted only if its holdout MAPE is
  no worse than the serving model's plus a small tolerance; otherwise the previous
  version keeps serving and the candidate is archived. Nothing worse ever goes live
  silently.
- **Semantic versions**: automatic retrains bump the minor version; `appraisenet serve`
  hot-reloads when `current/meta.json` changes, with zero downtime.
- **No private data required**: without `APPRAISENET_DB`, `train-production` fits on
  the synthetic corpus, so anyone cloning this repository can exercise the full
  registry, serving and monitoring loop end to end.
