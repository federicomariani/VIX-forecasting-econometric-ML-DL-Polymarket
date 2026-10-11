# Forecasting the VIX Index: A Multi-Horizon Benchmark AcrossEconometric, Machine Learning and Deep Learning Models withPrediction Market Signals

## Authors
- Federico Mariani
- Andrea Bucci
- Luca Romeo

## Abstract

This study compares forecasting models for the CBOE Volatility Index (VIX) belonging to four model families:

- **Random Walk** (benchmark)
- **Econometric** (ARIMA(1,1,1), HAR-Type)
- **Machine Learning** (Random Forest, XGBoost)
- **Deep Learning** (LSTM, Temporal Fusion Transformer)

The models are evaluated across three forecasting horizons (**1, 5, and 22 days**), using the same time period for all models.

An original contribution of this thesis is the integration of features derived from prediction markets (**Polymarket** - FOMC interest rate decisions, US recession probabilities, and year-over-year CPI inflation) into the econometric, Machine Learning, and Deep Learning models, in order to assess whether the probabilities implied by these markets provide additional useful information for forecasting. For each model, two runs are therefore performed: **“with PM”** and **“without PM”**, which are then compared against each other.

The models are optimized using both **Grid Search** and **Bayesian Optimization** and evaluated using point forecast accuracy metrics (MSE, MAE, MAPE, R², and QLIKE), Directional Accuracy, **Diebold–Mariano tests**, and the **Model Confidence Set (MCS)** procedure. The analysis is also conducted across different volatility regimes (**Low / Medium / High**).

## Repository Structure

```
├── 0_Dataset/
│   ├── Data/
│   │   ├── Raw/                                      # Raw macroeconomic time series from FRED (CSV)
│   │   │   ├── credit_spread.csv
│   │   │   ├── initial_claims.csv
│   │   │   ├── tbill_3m.csv
│   │   │   └── tyield_10y.csv
│   │   ├── Clean/
│   │   │   ├── dataset.csv                           # Final dataset used by models without Polymarket features
│   │   │   ├── dataset_adj_x_polymarket.csv          # Final dataset adjusted to match the length of the Polymarket dataset
│   │   │   └── Info.txt
│   │   └── download_data.py                          # Downloads VIX and explanatory variables from yfinance and FRED, then creates the features
│   │
│   └── Polymarket_data/
│       ├── Raw/                                      # Raw data downloaded from Polymarket public APIs, organized by event
│       │   ├── 0_FOMC_Interest_Rates/                # Separate subfolders for each FOMC meeting (2025–2026)
│       │   ├── 1_US_Recession/                       # US recession prediction markets for 2025 and 2026
│       │   └── 2_CPI_YoY/                            # Year-over-year CPI inflation prediction markets, organized by month
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
│   ├── 1_ARIMA(1,1,1)/
│   │   ├── arima.py                                  # ARIMA model without Polymarket features
│   │   └── arima_pm.py                               # ARIMA model with Polymarket features
│   └── 2_HAR-Type/
│       ├── har_type.py                               # HAR-Type model without Polymarket features
│       └── har_type_pm.py                            # HAR-Type model with Polymarket features
│
├── 3_Machine_Learning/
│   ├── 1_Random_Forest/
│   │   ├── random_forest.py                          # Random Forest using VIX lags of 1, 5, and 22 days
│   │   ├── random_forest_pm.py                       # Random Forest with Polymarket features
│   │   └── random_forest_v2.py                       # Random Forest variant using all VIX lags up to 22 days
│   └── 2_XGBoost/
│       ├── xgboost_model.py                          # XGBoost model without Polymarket features
│       ├── xgboost_pm.py                             # XGBoost model with Polymarket features
│       └── xgboost_v2.py                             # XGBoost variant using the alternative lag specification
│
├── 4_Deep_Learning/
│   ├── 1_LSTM/
│   │   ├── lstm.py                                   # LSTM model without Polymarket features
│   │   └── lstm_pm.py                                # LSTM model with Polymarket features
│   └── 2_TFT/
│       ├── tft.py                                    # Temporal Fusion Transformer implemented using pytorch-forecasting, without Polymarket features
│       └── tft_pm.py                                 # Temporal Fusion Transformer with Polymarket features
│
├── 5_Forecasts_&_Error_Metrics/                      # Stores forecasts, evaluation metrics, plots, and hyperparameter tuning results for each model
│   ├── Normale/                                      # Results from runs without Polymarket features
│   │   ├── Modelli_econometrici/{h1,h5,h22}/         # Econometric model results for the 1-, 5-, and 22-day forecasting horizons
│   │   ├── Machine_Learning/{h1,h5,h22}/[v2/]        # Machine Learning results, including alternative v2 specifications where applicable
│   │   └── Deep_Learning/{h1,h5,h22}/                # Deep Learning results for each forecasting horizon
│   └── With_Polymarket/                              # Results for runs with Polymarket features, organized using the same structure
│       ├── Modelli_econometrici/{h1,h5,h22}/         # Econometric model results for each forecasting horizon
│       ├── Machine_Learning/{h1,h5,h22}/             # Machine Learning results for each forecasting horizon
│       └── Deep_Learning/{h1,h5,h22}/                # Deep Learning results for each forecasting horizon
│
└── 6_Final_Results/                                  # Final comparative evaluation of all models
    ├── metriche.py                                   # Calculates MSE, MAE, MAPE, R², QLIKE, and Directional Accuracy
    ├── regimi_vol.py                                 # Evaluates model performance across Low, Medium, and High volatility regimes
    ├── mcs.py                                        # Implements the Model Confidence Set (MCS) procedure
    ├── Results/                                      # Stores standard evaluation outputs
    │   ├── stats_models_h*.png                       # Comparative model performance statistics for each forecasting horizon
    │   ├── mcs_h*.png                                # Model Confidence Set results for each forecasting horizon
    │   └── vol_regime_h*.csv                         # Evaluation metrics by volatility regime
    └── With_Polymarket/                              # Comparative evaluation incorporating Polymarket features
        ├── metriche_pm.py                            # Calculates evaluation metrics for models using Polymarket features
        ├── regimi_vol_pm.py                          # Evaluates performance across volatility regimes with Polymarket features
        ├── mcs_pm.py                                 # Applies the Model Confidence Set procedure to the Polymarket analysis
        ├── diebold_mariano.py                        # Performs Diebold–Mariano tests comparing forecasts with and without Polymarket features
        └── Results/                                  # Stores plots and statistical results from the Polymarket comparison
            ├── dm_h*.png                             # Diebold–Mariano test results for each forecasting horizon
            ├── mcs_mse_h*.png                        # MCS results based on MSE for each forecasting horizon
            ├── mcs_qlike_h*.png                      # MCS results based on QLIKE for each forecasting horizon
            └── stats_models_pm_h*.png                # Comparative model performance statistics including Polymarket features
```

Each `*_pm.py` script corresponds to its namesake script, trained with the additional Polymarket features. Each forecasting horizon folder (`h1`, `h5`, `h22`) contains the results of Grid Search (`*_gridsearch_*.csv`) and Bayesian Optimization (`*_bayesoptimization_*.csv`), together with cross-validation plots and forecast-versus-actual plots.

## Data

- **VIX and Market Regressors**: S&P 500 (log returns), WTI, OVX, DXY, OVX–VIX spread, VIX moving averages and lags, and 14-day RSI - downloaded from Yahoo Finance using `yfinance` (`download_data.py`).
- **Macroeconomic Variables from FRED**: 3-month Treasury Bill rate, 10-year Treasury yield (used to derive the yield curve), credit spread, and initial jobless claims (`initial_claims`) — stored as CSV files in `0_Dataset/Data/Raw/`.
- **Polymarket**: Market-implied probabilities from prediction markets covering three macroeconomic topics relevant to volatility - FOMC interest rate decisions, US recession probabilities, and year-over-year CPI inflation. The data are downloaded through Polymarket's public APIs (`download_data_polymarket.py`), cleaned in two stages (`clean_polymarket.py`, `clean_polymarket_2.py`), and transformed into model features (`feature_engineering_polymarket.py`).
- **Final Datasets**: The dataset containing only traditional features is `0_Dataset/Data/Clean/dataset.csv`, while the dataset augmented with Polymarket features is `0_Dataset/Polymarket_data/dataset_w_polymarket.csv`.

## Methodology

1. **Data Collection and Cleaning** (`0_Dataset/`).
2. **Modeling** across three forecasting horizons (1, 5, and 22 days), covering each of the four model families, with hyperparameter optimization performed using both **Grid Search** and **Bayesian Optimization** (via `scikit-optimize` / `optuna`).
3. For the econometric, Machine Learning, and Deep Learning models, each run is performed **with and without Polymarket features** to isolate their incremental informational contribution.
4. For Random Forest and XGBoost, a `v2` variant is also available, using all VIX lags up to 22 days instead of only the 1-, 5-, and 22-day lags.
5. **Final Comparative Evaluation** (`6_Final_Results/`): point forecast error metrics, Directional Accuracy, volatility regime analysis, Diebold–Mariano tests comparing the with-Polymarket and without-Polymarket model pairs, and the Model Confidence Set (MCS) procedure for comparing all models.

## Results

Detailed results - broken down by model, forecasting horizon (1, 5, and 22 days), and with or without Polymarket features - are presented in plots and CSV files in the `5_Forecasts_&_Error_Metrics/` and `6_Final_Results/` directories, and discussed in the paper.
