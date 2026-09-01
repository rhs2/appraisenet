# Model card: catboost

- family: boosting
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 11.584%, median APE 8.024%, R2(log) 0.925
- holdout: MAPE 11.582%, median APE 8.046%,
  R2(log) 0.925, within 10%: 58.181%
- 80% conformal interval on the holdout: coverage 80.0%,
  median width 35.5% of price
- fit time (protocol total): 367.0 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
