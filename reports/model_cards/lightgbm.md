# Model card: lightgbm

- family: boosting
- protocol: 1,500 listings, 1,259 train / 241 holdout,
  5-fold out-of-fold selection, holdout scored once (synthetic corpus)
- cross-validation: MAPE 9.427%, median APE 5.228%, R2(log) 0.979
- holdout: MAPE 9.338%, median APE 4.498%,
  R2(log) 0.976, within 10%: 68.05%
- 80% conformal interval on the holdout: coverage 82.2%,
  median width 32.3% of price
- fit time (protocol total): 22.7 s
- intended use: research comparison; the production configuration is documented in the README
- limitations: US market, $2,000-$100,000 price band, model year 1990+;
  no collector or exotic vehicles; asking prices, not transaction prices
