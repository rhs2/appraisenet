# Model card: embed_mlp

- family: deep tabular
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 12.194%, median APE 8.447%, R2(log) 0.917
- holdout: MAPE 11.944%, median APE 8.438%,
  R2(log) 0.92, within 10%: 56.666%
- 80% conformal interval on the holdout: coverage 79.9%,
  median width 37.2% of price
- fit time (protocol total): 41.9 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
