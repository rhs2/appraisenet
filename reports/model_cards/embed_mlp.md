# Model card: embed_mlp

- family: deep tabular
- protocol: 1,500 listings, 1,259 train / 241 holdout,
  5-fold out-of-fold selection, holdout scored once (synthetic corpus)
- cross-validation: MAPE 65.113%, median APE 67.519%, R2(log) 0.362
- holdout: MAPE 51.571%, median APE 50.708%,
  R2(log) 0.56, within 10%: 7.469%
- 80% conformal interval on the holdout: coverage 87.1%,
  median width 239.1% of price
- fit time (protocol total): 4.3 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
