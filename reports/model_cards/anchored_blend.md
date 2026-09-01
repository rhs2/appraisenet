# Model card: anchored_blend

- family: anchored
- protocol: 1,174,659 listings, 1,057,399 train / 117,260 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 8.201%, median APE 4.721%, R2(log) 0.943
- holdout: MAPE 8.236%, median APE 4.682%,
  R2(log) 0.942, within 10%: 78.516%
- 80% conformal interval on the holdout: coverage 80.1%,
  median width 21.1% of price
- fit time (protocol total): 12633.8 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
