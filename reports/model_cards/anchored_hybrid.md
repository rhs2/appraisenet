# Model card: anchored_hybrid

- family: anchored
- protocol: 1,174,659 listings, 1,057,399 train / 117,260 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 8.298%, median APE 4.765%, R2(log) 0.942
- holdout: MAPE 8.325%, median APE 4.725%,
  R2(log) 0.941, within 10%: 78.152%
- 80% conformal interval on the holdout: coverage 80.2%,
  median width 21.4% of price
- fit time (protocol total): 475.0 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
