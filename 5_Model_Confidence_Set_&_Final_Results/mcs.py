import pandas as pd
import numpy as np
from model_confidence_set import ModelConfidenceSet


# ---------------
# IMPORT DATASET
# ---------------
base_path = (r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\5. Model Confidence Set\+ Polymarkets\0_Forecast\h22")

files = {
    # Random Forest
    "Random Forest CV (con PM)": "random_forest_gridsearch_h22_con PM.csv",
    "Random Forest CV (senza PM)": "random_forest_gridsearch_h22_senza PM.csv",
    "Random Forest BO (con PM)": "random_forest_bayesoptimization_h22_con PM.csv",
    "Random Forest BO (senza PM)": "random_forest_bayesoptimization_h22_senza PM.csv",
    
    # XGBoost
    "XGBoost CV (con PM)": "xgboost_gridsearch_h22_con PM.csv",
    "XGBoost CV (senza PM)": "xgboost_gridsearch_h22_senza PM.csv",
    "XGBoost BO (con PM)": "xgboost_bayesoptimization_h22_con PM.csv",
    "XGBoost BO (senza PM)": "xgboost_bayesoptimization_h22_senza PM.csv",
    
    # LSTM
    "LSTM CV (con PM)": "lstm_gridsearch_h22_con PM.csv",
    "LSTM CV (senza PM)": "lstm_gridsearch_h22_senza PM.csv",
    "LSTM BO (con PM)": "lstm_bayesoptimization_h22_con PM.csv",
    "LSTM BO (senza PM)": "lstm_bayesoptimization_h22_senza PM.csv",
    
    # TFT
    "TFT CV (con PM)": "tft_gridsearch_h22_con PM.csv",
    "TFT CV (senza PM)": "tft_gridsearch_h22_senza PM.csv",
    "TFT BO (con PM)": "tft_bayesoptimization_h22_con PM.csv",
    "TFT BO (senza PM)": "tft_bayesoptimization_h22_senza PM.csv"
}



# -------------------------------
# FUNZIONI DI LOSS (QLIKE & MSE)
# -------------------------------
def compute_qlike(actual, forecast): 
    return (actual / forecast) - np.log(actual / forecast) - 1


def compute_mse(actual, forecast):
    return (actual - forecast) ** 2


# -----------------------------------------------------------
# CALCOLO FUNZIONI LOSS PER OGNI FILE CSV (PER OGNI MODELLO)
# -----------------------------------------------------------
mse_series = {}
qlike_series = {}

for model_name, filename in files.items():
    path = rf"{base_path}\{filename}"
    df = pd.read_csv(path)
    df = pd.read_csv(path).dropna()

    df["Loss_mse"] = compute_mse(df["Actual"], df["Forecast"])
    df["Loss_qlike"] = compute_qlike(df["Actual"], df["Forecast"])

    df = df.set_index("Date")
    mse_series[model_name] = df["Loss_mse"]
    qlike_series[model_name] = df["Loss_qlike"]


# -----------------------------------------------------------
# COSTRUZIONE DATAFRAME SEPARATI PER LE LOSSES (QLIKE E MSE)
# -----------------------------------------------------------
losses_mse = pd.concat(mse_series, axis=1).dropna() # Uso indice Date per unire e poi lo droppo in quanto richiesto dal MCS
losses_qlike = pd.concat(qlike_series, axis=1).dropna()


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


# -----------------------------------------------------------------
# DIAGNOSTICA OUTLIER E ALTRO PER VERIFICARE LEGITTIMITÀ RISULTATI
# -----------------------------------------------------------------
print("\n=== Diagnostica MSE (statistica descrittiva errori) ===")
print(losses_mse.describe().T[["mean", "std", "min", "max"]])
 
print("\n=== Diagnostica QLIKE (statistica descrittiva errori) ===")
print(losses_qlike.describe().T[["mean", "std", "min", "max"]])
 
print("\n=== Forecast <= 0 per modello (problematico per QLIKE) ===")
for model_name, filename in files.items():
    path = rf"{base_path}\{filename}"
    df = pd.read_csv(path)
    n_bad = (df["Forecast"] <= 0).sum()
    if n_bad > 0:
        print(f"{model_name}: {n_bad} forecast <= 0")