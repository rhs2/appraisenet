# Model card: extra_trees

- family: bagged trees
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 13.108%, median APE 8.923%, R2(log) 0.904
- holdout: MAPE 12.877%, median APE 8.543%,
  R2(log) 0.905, within 10%: 55.613%
- 80% conformal interval on the holdout: coverage 79.9%,
  median width 39.9% of price
- fit time (protocol total): 17.3 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
