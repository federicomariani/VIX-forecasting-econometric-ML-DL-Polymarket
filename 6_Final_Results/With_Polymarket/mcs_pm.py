import pandas as pd
import numpy as np
from pathlib import Path
from model_confidence_set import ModelConfidenceSet


# ---------------
# IMPORT DATASET
# ---------------
orizzonte = 22  # Da cambiare ad ogni run (1, 5 e 22)

# Cambia da solo il path quando si cambia orizzonte
econometric_path = Path(
    rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Modelli_econometrici\h{orizzonte}"
)

ml_path = Path(
    rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Machine_Learning\h{orizzonte}"
)

dl_path = Path(
    rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Deep_Learning\h{orizzonte}"
)

files = {
    # ARIMA(1,1,1)
    "ARIMA(1,1,1) (con PM)": (econometric_path, f"arima_h{orizzonte}_con_PM.csv"),
    "ARIMA(1,1,1) (senza PM)": (econometric_path, f"arima_h{orizzonte}_senza_PM.csv"),

    # HAR-Type
    "HAR-Type (con PM)": (econometric_path, f"har_h{orizzonte}_con_PM.csv"),
    "HAR-Type (senza PM)": (econometric_path, f"har_h{orizzonte}_senza_PM.csv"),

    # Random Forest
    "Random Forest CV (con PM)": (ml_path, f"random_forest_gridsearch_h{orizzonte}_con_PM.csv"),
    "Random Forest CV (senza PM)": (ml_path, f"random_forest_gridsearch_h{orizzonte}_senza_PM.csv"),
    "Random Forest BO (con PM)": (ml_path, f"random_forest_bayesoptimization_h{orizzonte}_con_PM.csv"),
    "Random Forest BO (senza PM)": (ml_path, f"random_forest_bayesoptimization_h{orizzonte}_senza_PM.csv"),

    # XGBoost
    "XGBoost CV (con PM)": (ml_path, f"xgboost_gridsearch_h{orizzonte}_con_PM.csv"),
    "XGBoost CV (senza PM)": (ml_path, f"xgboost_gridsearch_h{orizzonte}_senza_PM.csv"),
    "XGBoost BO (con PM)": (ml_path, f"xgboost_bayesoptimization_h{orizzonte}_con_PM.csv"),
    "XGBoost BO (senza PM)": (ml_path, f"xgboost_bayesoptimization_h{orizzonte}_senza_PM.csv"),

    # LSTM
    "LSTM CV (con PM)": (dl_path, f"lstm_gridsearch_h{orizzonte}_con_PM.csv"),
    "LSTM CV (senza PM)": (dl_path, f"lstm_gridsearch_h{orizzonte}_senza_PM.csv"),
    "LSTM BO (con PM)": (dl_path, f"lstm_bayesoptimization_h{orizzonte}_con_PM.csv"),
    "LSTM BO (senza PM)": (dl_path, f"lstm_bayesoptimization_h{orizzonte}_senza_PM.csv"),

    # TFT
    "TFT CV (con PM)": (dl_path, f"tft_gridsearch_h{orizzonte}_con_PM.csv"),
    "TFT CV (senza PM)": (dl_path, f"tft_gridsearch_h{orizzonte}_senza_PM.csv"),
    "TFT BO (con PM)": (dl_path, f"tft_bayesoptimization_h{orizzonte}_con_PM.csv"),
    "TFT BO (senza PM)": (dl_path, f"tft_bayesoptimization_h{orizzonte}_senza_PM.csv"),
}

available_files = {
    model_name: (folder, filename)
    for model_name, (folder, filename) in files.items()
    if (folder / filename).exists()
}

if not available_files:
    raise FileNotFoundError(
        "Nessun file CSV trovato per l'orizzonte selezionato. "
        "Controlla i path delle cartelle e i nomi dei dataset."
    )

missing_files = set(files) - set(available_files)
if missing_files:
    print("\n=== FILE MANCANTI (ignorati) ===")
    for model_name in sorted(missing_files):
        folder, filename = files[model_name]
        print(f"- {model_name}: {folder / filename}")


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

for model_name, (folder, filename) in available_files.items():
    path = folder / filename
    df = pd.read_csv(path).dropna()

    if {"Date", "Actual", "Forecast"}.difference(df.columns):
        raise ValueError(f"Il file {path.name} non contiene le colonne attese: Date, Actual, Forecast")

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df["Loss_mse"] = compute_mse(df["Actual"], df["Forecast"])
    df["Loss_qlike"] = compute_qlike(df["Actual"], df["Forecast"])
    df = df.set_index("Date")

    mse_series[model_name] = df["Loss_mse"]
    qlike_series[model_name] = df["Loss_qlike"]


# -----------------------------------------------------------
# COSTRUZIONE DATAFRAME SEPARATI PER LE LOSSES (QLIKE E MSE)
# -----------------------------------------------------------
losses_mse = pd.concat(mse_series, axis=1).dropna()
losses_qlike = pd.concat(qlike_series, axis=1).dropna()

if losses_mse.empty or losses_qlike.empty:
    raise ValueError(
        "Non ci sono osservazioni comuni tra i forecast. "
        "Controllare i formati delle colonne Date e i periodi OOS."
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


# -----------------------------------------------------------------
# DIAGNOSTICA OUTLIER E ALTRO PER VERIFICARE LEGITTIMITÀ RISULTATI
# -----------------------------------------------------------------
print("\n=== Diagnostica MSE (statistica descrittiva errori) ===")
print(losses_mse.describe().T[["mean", "std", "min", "max"]])

print("\n=== Diagnostica QLIKE (statistica descrittiva errori) ===")
print(losses_qlike.describe().T[["mean", "std", "min", "max"]])

print("\n=== Forecast <= 0 per modello (problematico per QLIKE) ===")
for model_name, (folder, filename) in available_files.items():
    path = folder / filename
    df = pd.read_csv(path)
    n_bad = (df["Forecast"] <= 0).sum()
    if n_bad > 0:
        print(f"{model_name}: {n_bad} forecast <= 0")