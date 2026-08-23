# Model card: random_forest

- family: bagged trees
- protocol: 1,500 listings, 1,259 train / 241 holdout,
  5-fold out-of-fold selection, holdout scored once (synthetic corpus)
- cross-validation: MAPE 10.193%, median APE 6.398%, R2(log) 0.975
- holdout: MAPE 8.951%, median APE 4.477%,
  R2(log) 0.977, within 10%: 68.465%
- 80% conformal interval on the holdout: coverage 83.4%,
  median width 35.3% of price
- fit time (protocol total): 1.0 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
