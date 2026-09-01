# Model card: anchored_lgbm

- family: anchored
- protocol: 1,174,659 listings, 1,057,399 train / 117,260 holdout,
  5-fold out-of-fold selection, holdout scored once (private dataset)
- cross-validation: MAPE 7.313%, median APE 4.748%, R2(log) 0.959
- holdout: MAPE 7.314%, median APE 4.754%,
  R2(log) 0.959, within 10%: 78.604%
- 80% conformal interval on the holdout: coverage 80.1%,
  median width 21.0% of price
- fit time (protocol total): 437.7 s
- deployment: this is the configuration `appraisenet train-production` fits and serves
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
