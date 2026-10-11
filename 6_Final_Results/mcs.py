import pandas as pd
import numpy as np
from pathlib import Path
from model_confidence_set import ModelConfidenceSet



# ---------------
# IMPORT DATASET
# ---------------
orizzonte = 22 # Change on each run (1, 5, or 22)

# Automatically update the path when the forecast horizon changes
ml_path = Path(
    rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\Normale\Machine_Learning\h{orizzonte}"
)
dl_path = Path(
    rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\Normale\Deep_Learning\h{orizzonte}"
)
econometric_path = Path(
    rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\Normale\Modelli_econometrici\h{orizzonte}"
)

files = {
    # Stochastic and econometric models
    "Random Walk": (econometric_path, f"randomwalk_h{orizzonte}.csv"),
    "ARIMA": (econometric_path, f"arima_h{orizzonte}.csv"),
    "HAR": (econometric_path, f"har_h{orizzonte}.csv"),

    # Random Forest 
    "Random Forest CV": (ml_path, f"random_forest_gridsearch_h{orizzonte}.csv"),
    "Random Forest BO": (ml_path, f"random_forest_bayesoptimization_h{orizzonte}.csv"),
    
    # XGBoost 
    "XGBoost CV": (ml_path, f"xgboost_gridsearch_h{orizzonte}.csv"),
    "XGBoost BO": (ml_path, f"xgboost_bayesoptimization_h{orizzonte}.csv"),
    
    # LSTM 
    "LSTM CV": (dl_path, f"lstm_gridsearch_h{orizzonte}.csv"),
    "LSTM BO": (dl_path, f"lstm_bayesoptimization_h{orizzonte}.csv"),
    
    # TFT
    "TFT CV": (dl_path, f"tft_gridsearch_h{orizzonte}.csv"),
    "TFT BO": (dl_path, f"tft_bayesoptimization_h{orizzonte}.csv")
}


# -----------------------------
# LOSS FUNCTIONS (QLIKE & MSE)
# -----------------------------
def compute_qlike(actual, forecast): 
    return (actual / forecast) - np.log(actual / forecast) - 1


def compute_mse(actual, forecast):
    return (actual - forecast) ** 2


# ----------------------------------------------------------
# COMPUTE LOSS FUNCTIONS FOR EACH CSV FILE (FOR EACH MODEL)
# ----------------------------------------------------------
mse_series = {}
qlike_series = {}

for model_name, (folder, filename) in files.items():
    path = folder / filename
    df = pd.read_csv(path).dropna()

    df["Loss_mse"] = compute_mse(df["Actual"], df["Forecast"])
    df["Loss_qlike"] = compute_qlike(df["Actual"], df["Forecast"])

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df.set_index("Date")
    mse_series[model_name] = df["Loss_mse"]
    qlike_series[model_name] = df["Loss_qlike"]


# ---------------------------------------------------------------------
# BUILD SEPARATE DATAFRAMES FOR THE TWO LOSS FUNCTIONS (QLIKE AND MSE)
# ---------------------------------------------------------------------
losses_mse = pd.concat(mse_series, axis=1).dropna() # Use Date as the index to merge, then drop it because MCS requires it
losses_qlike = pd.concat(qlike_series, axis=1).dropna()

if losses_mse.empty or losses_qlike.empty:
    raise ValueError(
        "There are no common observations across the forecasts. "
        "Check the Date column formats and the OOS periods."
    )


# ---------------------
# MODEL CONFIDENCE SET
# ---------------------
def run_mcs(losses, label):
    print(f"\n=== Model Confidence Set - {label} ===")
    mcs = ModelConfidenceSet(
        losses,
        n_boot=5000,
        alpha=0.05,
        method="R",
    )
    mcs.compute()
    results = mcs.results()
    print(results)
    return results


results_mse = run_mcs(losses_mse, "MSE")
results_qlike = run_mcs(losses_qlike, "QLIKE")


# ---------------------------------------------------------
# OUTLIER DIAGNOSTICS AND OTHER CHECKS TO VALIDATE RESULTS
# ---------------------------------------------------------
print("\n=== MSE Diagnostic (descriptive statistics of errors) ===")
print(losses_mse.describe().T[["mean", "std", "min", "max"]])
 
print("\n=== QLIKE Diagnostic (descriptive statistics of errors) ===")
print(losses_qlike.describe().T[["mean", "std", "min", "max"]])
 
print("\n=== Forecast <= 0 by model (problematic for QLIKE) ===")
for model_name, (folder, filename) in files.items():
    path = folder / filename
    df = pd.read_csv(path)
    n_bad = (df["Forecast"] <= 0).sum()
    if n_bad > 0:
        print(f"{model_name}: {n_bad} forecast <= 0")