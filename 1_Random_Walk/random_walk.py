import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



# -----------------
# DOWNLOAD DATASET
# -----------------
dataset = yf.download("^VIX", start = "2014-02-01", end = "2026-07-01")


# ------------------
# PREPARAZIONE DATI
# ------------------
dataset = dataset[["Close"]]

if isinstance(dataset.columns, pd.MultiIndex):
    dataset.columns = dataset.columns.droplevel(1)

dataset = dataset.dropna(subset=["Close"])

dataset = dataset.sort_index()

# Split dataset
oos_start = pd.Timestamp("2022-10-05")
oos_end = pd.Timestamp("2026-06-30")


# -------------------
# CARTELLA RISULTATI
# -------------------
# Percorso relativo alla radice della repo, indipendente dal PC su cui gira lo script.
# I CSV finiscono in .../Modelli_econometrici/h{h}/, dove li leggono mcs.py e diebold_mariano.py.
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(repo_dir, "5_Forecasts_&_Error_Metrics", "Normale", "Modelli_econometrici")

# False = stampa solo le metriche, senza sovrascrivere le previsioni salvate
# (i dati scaricati oggi da Yahoo Finance possono differire leggermente da quelli originali).
SALVA_CSV = False


# --------------------
# MODELLO RANDOM WALK
# --------------------
def run_random_walk(h):
    dataset_rw = dataset.copy()

    dataset_rw["y_true"] = dataset_rw["Close"]
    dataset_rw["y_pred"] = (dataset_rw["Close"].shift(h))

    # Selezione periodo OOS 
    results = dataset_rw.loc[(dataset_rw.index >= oos_start) & (dataset_rw.index <= oos_end)].copy()

    # Elimina NaN
    results = results.dropna(subset=["y_true", "y_pred"])

    # ---------
    # METRICHE
    # ---------
    mse = mean_squared_error(results["y_true"], results["y_pred"])
    mae = mean_absolute_error(results["y_true"], results["y_pred"])
    mape = (mean_absolute_percentage_error(results["y_true"], results["y_pred"]) * 100)
    qlike = np.mean((results["y_true"] / results["y_pred"]) - np.log(results["y_true"] / results["y_pred"]) - 1)

    # R^2 out-of-sample, calcolato come per gli altri modelli: 1 - SSE/SST
    ss_res = np.sum((results["y_true"] - results["y_pred"]) ** 2)
    ss_tot = np.sum((results["y_true"] - results["y_true"].mean()) ** 2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # R^2 di Mincer-Zarnowitz (MZ), solo come diagnostica: non è confrontabile con l'R^2 OOS
    X = sm.add_constant(results["y_pred"], has_constant="add")
    model_mz = sm.OLS(results["y_true"], X).fit()
    r2_mz = model_mz.rsquared
    
    # Directional Accuracy rispetto al VIX osservato alla data d'origine.
    origin_positions = dataset.index.get_indexer(results.index) - h
    if (origin_positions < 0).any():
        raise ValueError(f"Data d'origine non disponibile per il random walk h={h}.")
    origin_vix = dataset["Close"].iloc[origin_positions].to_numpy()
    actual_direction = results["y_true"].to_numpy() > origin_vix
    # Il random walk senza drift prevede V_{t+h|t} = V_t, quindi
    # l'indicatore previsto I(V_hat_{t+h|t} > V_t) è sempre False.
    predicted_direction = results["y_pred"].to_numpy() > origin_vix
    directional_accuracy = (actual_direction == predicted_direction).mean() * 100

    # ----------
    # RISULTATI
    # ----------
    print(f"MSE h_{h}: {mse:.6f}")
    print(f"MAE h_{h}: {mae:.6f}")
    print(f"MAPE h_{h}: {mape:.6f}%")
    print(f"QLIKE h_{h}: {qlike:.6f}")
    print(f"R^2 OOS h_{h}: {r2_oos:.6f}")
    print(f"R^2 MZ h_{h} (diagnostica): {r2_mz:.6f}")
    print(f"Directional Accuracy h_{h}: {directional_accuracy:.2f}%")

    # File CSV contenente forecasts
    df_out = pd.DataFrame({
        "Actual": results["y_true"].values,
        "Forecast": results["y_pred"].values,
        "Origin_VIX": origin_vix,
    })

    df_out.index = results.index
    df_out.index.name = "Date"

    if SALVA_CSV:
        h_dir = os.path.join(output_dir, f"h{h}")
        os.makedirs(h_dir, exist_ok=True)
        df_out.to_csv(os.path.join(h_dir, f"randomwalk_h{h}.csv"))


results_rw_h1 = run_random_walk(1)
results_rw_h5 = run_random_walk(5)
results_rw_h22 = run_random_walk(22)
