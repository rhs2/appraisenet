# Model card: elasticnet

- family: linear
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 19.319%, median APE 14.276%, R2(log) 0.813
- holdout: MAPE 19.418%, median APE 14.433%,
  R2(log) 0.807, within 10%: 36.501%
- 80% conformal interval on the holdout: coverage 79.8%,
  median width 60.2% of price
- fit time (protocol total): 6.6 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
