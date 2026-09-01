# Model card: ft_transformer

- family: deep tabular
- protocol: 38,758 listings, 34,865 train / 3,893 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 12.76%, median APE 9.185%, R2(log) 0.91
- holdout: MAPE 12.357%, median APE 8.897%,
  R2(log) 0.915, within 10%: 54.688%
- 80% conformal interval on the holdout: coverage 80.9%,
  median width 39.9% of price
- fit time (protocol total): 798.8 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
