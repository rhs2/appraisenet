# Model card: hybrid_lgbm_text

- family: hybrid
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 10.906%, median APE 7.412%, R2(log) 0.931
- holdout: MAPE 10.754%, median APE 7.335%,
  R2(log) 0.931, within 10%: 62.445%
- 80% conformal interval on the holdout: coverage 79.8%,
  median width 33.3% of price
- fit time (protocol total): 327.9 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
