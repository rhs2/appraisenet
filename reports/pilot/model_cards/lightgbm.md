# Model card: lightgbm

- family: boosting
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 11.399%, median APE 7.792%, R2(log) 0.926
- holdout: MAPE 11.243%, median APE 7.631%,
  R2(log) 0.926, within 10%: 60.622%
- 80% conformal interval on the holdout: coverage 79.6%,
  median width 34.9% of price
- fit time (protocol total): 101.1 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
