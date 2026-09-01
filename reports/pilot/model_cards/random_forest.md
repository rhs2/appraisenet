# Model card: random_forest

- family: bagged trees
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 13.733%, median APE 9.42%, R2(log) 0.896
- holdout: MAPE 13.495%, median APE 9.243%,
  R2(log) 0.898, within 10%: 53.044%
- 80% conformal interval on the holdout: coverage 80.1%,
  median width 41.9% of price
- fit time (protocol total): 22.9 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
