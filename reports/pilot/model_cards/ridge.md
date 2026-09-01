# Model card: ridge

- family: linear
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 18.897%, median APE 14.071%, R2(log) 0.821
- holdout: MAPE 18.935%, median APE 14.178%,
  R2(log) 0.817, within 10%: 37.683%
- 80% conformal interval on the holdout: coverage 80.1%,
  median width 58.7% of price
- fit time (protocol total): 5.5 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
