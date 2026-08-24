# Model card: stack

- family: hybrid
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 11.156%, median APE 7.553%, R2(log) 0.928
- holdout: MAPE 11.019%, median APE 7.577%,
  R2(log) 0.928, within 10%: 61.058%
- 80% conformal interval on the holdout: coverage 79.7%,
  median width 34.3% of price
- fit time (protocol total): 858.4 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
