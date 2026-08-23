# Model card: ridge

- family: linear
- protocol: 1,500 listings, 1,259 train / 241 holdout,
  5-fold out-of-fold selection, holdout scored once (synthetic corpus)
- cross-validation: MAPE 26.671%, median APE 23.981%, R2(log) 0.895
- holdout: MAPE 25.014%, median APE 22.412%,
  R2(log) 0.903, within 10%: 22.822%
- 80% conformal interval on the holdout: coverage 78.4%,
  median width 81.5% of price
- fit time (protocol total): 1.0 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
