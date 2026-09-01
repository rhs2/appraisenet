# Model card: knn_comparables

- family: instance
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 15.537%, median APE 11.127%, R2(log) 0.873
- holdout: MAPE 15.216%, median APE 10.788%,
  R2(log) 0.877, within 10%: 47.007%
- 80% conformal interval on the holdout: coverage 80.5%,
  median width 47.9% of price
- fit time (protocol total): 6.6 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
