# Model card: blend_lgbm_catboost

- family: hybrid
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 11.143%, median APE 7.592%, R2(log) 0.929
- holdout: MAPE 11.081%, median APE 7.481%,
  R2(log) 0.929, within 10%: 60.57%
- 80% conformal interval on the holdout: coverage 79.9%,
  median width 34.0% of price
- fit time (protocol total): 471.5 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
