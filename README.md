# Forecasting the VIX: Do Machine Learning and Prediction Market Signals Improve Predictive Accuracy?

## Authors
- Federico Mariani
- Andrea Bucci
- Emanuela Giacomini
- Luca Romeo

## Abstract

This study compares forecasting models for the CBOE Volatility Index (VIX) belonging to four model families:

- **Random Walk** (benchmark)
- **Econometric** (HAR and HAR-X)
- **Machine Learning** (Random Forest, XGBoost)
- **Deep Learning** (LSTM, Temporal Fusion Transformer)

The models are evaluated across three forecasting horizons (**1, 5, and 22 days**), using the same time period for all models.

An original contribution of this paper is the integration of features derived from prediction markets (**Polymarket** - FOMC interest rate decisions, US recession probabilities, and year-over-year CPI inflation) into the econometric, Machine Learning, and Deep Learning models, in order to assess whether the probabilities implied by these markets provide additional useful information for forecasting. For each model, two runs are therefore performed: **"with PM"** and **"without PM"**, which are then compared against each other.

The models are optimized using both **Grid Search** and **Bayesian Optimization** and evaluated using point forecast accuracy metrics (MSE, MAE, MAPE, R², and QLIKE), Directional Accuracy, **Diebold–Mariano tests** with Holm and Benjamini–Hochberg adjustments, and the **Model Confidence Set (MCS)** procedure. The analysis is also conducted across different volatility regimes (**Low / Medium / High**).

## Version 3 (results in the paper)

The results reported in the paper come from the `*_v3.py` scripts. Compared with the first version:

- **Common information set.** HAR-X, Random Forest, XGBoost, LSTM and TFT all use the same predictors: the HAR components of the VIX at the forecast origin (V_t, its 5-day and 22-day averages) plus the financial, energy and macroeconomic variables. In the Polymarket analysis the Polymarket features are added to this set. The weekday dummy is no longer used.
- **HAR-X.** New script `har_type_v3.py` (and `har_type_pm_v3.py`): the HAR model augmented with the same exogenous variables, estimated by OLS on the same rolling window.
- **Deep Learning.** Lookback of 5 trading days. The first version of the Polymarket LSTM/TFT scripts did not include `Close_VIX` among the inputs; this is fixed in the v3 scripts.
- **ARIMA removed.** The ARIMA(1,1,1) model and its outputs are no longer part of the comparison.
- **Evaluation.** The scripts `mcs.py`, `regimi_vol.py` and `With_Polymarket/*.py` of the first version are replaced by `6_Final_Results/evaluation_v3.py` (metrics, regimes, MCS, Diebold–Mariano tests) and `6_Final_Results/make_tables_v3.py` (LaTeX tables and regime figures).

The v3 scripts write their outputs in a `v3/` subfolder of each horizon folder, so the first-version outputs are kept. The Random Walk (`randomwalk_h*.csv`) and the plain HAR (`har_h*.csv`) forecasts used in the paper are those produced by `random_walk.py` and `har_type.py` / `har_type_pm.py`.

## Repository Structure

```
├── 0_Dataset/
│   ├── Data/
│   │   ├── Raw/                                      # Raw macroeconomic time series from FRED (CSV)
│   │   ├── Clean/
│   │   │   ├── dataset.csv                           # Final dataset used by models without Polymarket features
│   │   │   ├── dataset_adj_x_polymarket.csv          # Final dataset adjusted to match the length of the Polymarket dataset
│   │   │   └── Info.txt
│   │   └── download_data.py                          # Downloads VIX and explanatory variables from yfinance and FRED, then creates the features
│   │
│   └── Polymarket_data/
│       ├── Raw/                                      # Raw data downloaded from Polymarket public APIs, organized by event
│       ├── Clean/                                    # Cleaned Polymarket data, including the aggregated df_polymarket.csv file
│       ├── dataset_w_polymarket.csv                  # Final dataset combining traditional features with Polymarket-derived features
│       ├── download_data_polymarket.py               # Downloads data through Polymarket's public APIs
│       ├── clean_polymarket.py                       # First stage of Polymarket data cleaning
│       ├── clean_polymarket_2.py                     # Second stage of Polymarket data cleaning
│       ├── feature_engineering_polymarket.py         # Constructs features from prediction market-implied probabilities
│       └── Info.txt
│
├── 1_Random_Walk/
│   └── random_walk.py                                # Random Walk benchmark model
│
├── 2_Econometrics/
│   └── 2_HAR-Type/
│       ├── har_type.py                               # HAR model without Polymarket features
│       ├── har_type_pm.py                            # HAR model with Polymarket features
│       ├── har_type_v3.py                            # HAR-X (v3) without Polymarket features
│       └── har_type_pm_v3.py                         # HAR-X (v3) with Polymarket features
│
├── 3_Machine_Learning/
│   ├── 1_Random_Forest/
│   │   ├── random_forest.py / random_forest_pm.py    # First version
│   │   ├── random_forest_v2.py                       # Variant using the 22 most recent daily VIX values
│   │   ├── random_forest_v3.py                       # v3, without Polymarket features
│   │   └── random_forest_pm_v3.py                    # v3, with Polymarket features
│   └── 2_XGBoost/
│       ├── xgboost_model.py / xgboost_pm.py          # First version
│       ├── xgboost_v2.py                             # Variant using the 22 most recent daily VIX values
│       ├── xgboost_v3.py                             # v3, without Polymarket features
│       └── xgboost_pm_v3.py                          # v3, with Polymarket features
│
├── 4_Deep_Learning/
│   ├── 1_LSTM/
│   │   ├── lstm.py / lstm_pm.py                      # First version
│   │   ├── lstm_v3.py                                # v3, without Polymarket features
│   │   └── lstm_pm_v3.py                             # v3, with Polymarket features
│   └── 2_TFT/
│       ├── tft.py / tft_pm.py                        # First version (pytorch-forecasting)
│       ├── tft_v3.py                                 # v3, without Polymarket features
│       └── tft_pm_v3.py                              # v3, with Polymarket features
│
├── 5_Forecasts_&_Error_Metrics/                      # Forecasts, evaluation metrics, plots and hyperparameter tuning results
│   ├── Normale/                                      # Runs without Polymarket features
│   │   ├── Modelli_econometrici/{h1,h5,h22}/[v3/]
│   │   ├── Machine_Learning/{h1,h5,h22}/[v2/,v3/]
│   │   └── Deep_Learning/{h1,h5,h22}/[v3/]
│   └── With_Polymarket/                              # Runs with Polymarket features, same structure
│       ├── Modelli_econometrici/{h1,h5,h22}/[v3/]
│       ├── Machine_Learning/{h1,h5,h22}/[v3/]
│       └── Deep_Learning/{h1,h5,h22}/[v3/]
│
└── 6_Final_Results/
    ├── evaluation_v3.py                              # Metrics, volatility regimes, MCS and Diebold–Mariano tests (v3)
    ├── make_tables_v3.py                             # LaTeX tables and regime figures from the outputs of evaluation_v3.py
    └── Results/v3/
        ├── metrics_full.csv                          # MSE, MAE, MAPE, R², QLIKE and DA, full sample
        ├── regimes_full.csv                          # Metrics by volatility regime, full sample
        ├── mcs_full.csv                              # MCS p-values, full sample (MSE and QLIKE)
        ├── dm_pm.csv                                 # Diebold–Mariano tests, with vs without Polymarket, with Holm and BH adjustments
        ├── losses_pm.csv                             # MSE and QLIKE, with vs without Polymarket
        ├── mcs_pm.csv                                # MCS p-values, Polymarket sample
        ├── regimes_pm.csv                            # Metrics by volatility regime, Polymarket sample
        ├── tables_v3.tex                             # Tables of the paper
        └── vol_regimes_v3.png, vol_regimes_pm_v3.png # Regime figures of the paper
```

In the Polymarket folders, file names end in `_con_PM` (with Polymarket features) or `_senza_PM` (same sample, without Polymarket features). In the v3 Machine Learning Polymarket folders the Bayesian Optimization files are named `*_bayesopt_*` instead of `*_bayesoptimization_*`, to keep the full paths below the Windows limit of 260 characters.

## How to run (v3)

1. `0_Dataset/`: build the datasets (only needed to update the data).
2. `1_Random_Walk/random_walk.py`, `2_Econometrics/2_HAR-Type/har_type.py` and `har_type_pm.py`: Random Walk and HAR benchmarks.
3. The `*_v3.py` scripts in `2_Econometrics/`, `3_Machine_Learning/` and `4_Deep_Learning/`. The Machine Learning and Deep Learning scripts take the horizon as first argument, and the Polymarket ones the sample as second argument, e.g. `python random_forest_pm_v3.py 5 senza_PM`; `har_type_v3.py` runs all horizons, and `har_type_pm_v3.py` takes only the sample (`con_PM` or `senza_PM`). The Polymarket scripts are run for both `con_PM` and `senza_PM`.
4. `6_Final_Results/evaluation_v3.py`, then `6_Final_Results/make_tables_v3.py`.

All paths are relative to the repository root.

Main packages: `numpy`, `pandas`, `statsmodels`, `scikit-learn`, `scikit-optimize`, `boruta`, `xgboost`, `shap`, `torch`, `skorch`, `lightning`, `pytorch-forecasting`, `optuna`, `matplotlib`; for the evaluation also `dieboldmariano`, `model-confidence-set` and `numba`.

## Data

- **VIX and Market Regressors**: S&P 500 (log returns), WTI, OVX, DXY, OVX–VIX spread, VIX moving averages and lags, and 14-day RSI - downloaded from Yahoo Finance using `yfinance` (`download_data.py`).
- **Macroeconomic Variables from FRED**: 3-month Treasury Bill rate, 10-year Treasury yield (used to derive the yield curve), credit spread, and initial jobless claims (`initial_claims`) — stored as CSV files in `0_Dataset/Data/Raw/`.
- **Polymarket**: Market-implied probabilities from prediction markets covering three macroeconomic topics relevant to volatility - FOMC interest rate decisions, US recession probabilities, and year-over-year CPI inflation. The data are downloaded through Polymarket's public APIs (`download_data_polymarket.py`), cleaned in two stages (`clean_polymarket.py`, `clean_polymarket_2.py`), and transformed into model features (`feature_engineering_polymarket.py`).
- **Final Datasets**: The dataset containing only traditional features is `0_Dataset/Data/Clean/dataset.csv`, while the dataset augmented with Polymarket features is `0_Dataset/Polymarket_data/dataset_w_polymarket.csv`.
