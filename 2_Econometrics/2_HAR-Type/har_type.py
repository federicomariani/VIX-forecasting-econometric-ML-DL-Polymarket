import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



# -------------------------
# IMPORT DATASET E PULIZIA 
# -------------------------
dataset = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\Repo\2_Econometrics\2_HAR-Type\dataset_econometrics.csv")

dataset["Date"] = pd.to_datetime(dataset["Date"], dayfirst=True, errors="coerce")
dataset = (dataset.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True))


# ---------------------------------------
# COSTRUZIONE VARIABILI MODELLO HAR-TYPE
# ---------------------------------------
# Componente giornaliera
dataset["logVIX_d"] = dataset["VIX_log"]

# Componente settimanale
dataset["VIX_w"] = (dataset["VIX"].rolling(window=5).mean())
dataset["logVIX_w"] = np.log(dataset["VIX_w"])

# Componente mensile
dataset["VIX_m"] = (dataset["VIX"].rolling(window=22).mean())
dataset["logVIX_m"] = np.log(dataset["VIX_m"])

# Elimina NaN generati dalle medie
dataset = (dataset.dropna(subset=["logVIX_d", "logVIX_w", "logVIX_m"]).reset_index(drop=True))

# Features del modello
features = ["logVIX_d", "logVIX_w", "logVIX_m"]


# ----------------------------------
# PERIODO IN-SAMPLE E OUT-OF-SAMPLE
# ----------------------------------
is_start = pd.Timestamp("2014-01-01")
is_end = pd.Timestamp("2021-12-31")

oos_start = pd.Timestamp("2022-01-01")
oos_end = pd.Timestamp("2026-06-30")


# ------------------------------------
# CREAZIONE WINDOW PER ROLLING WINDOW
# ------------------------------------
dataset["Date_num"] = np.arange(1, len(dataset) + 1)

d1 = dataset.loc[dataset["Date"] == is_end, "Date_num"].iloc[0]
d2 = dataset.loc[dataset["Date"] == pd.Timestamp("2018-01-02"), "Date_num"].iloc[0]

window_size = d1 - d2 + 1


# ------------------------
# FUNZIONE HAR-TYPE MODEL
# ------------------------
def run_har_vix(h):
    target = f"VIX_target_h{h}"

    dataset[target] = dataset["VIX_log"].shift(-h)

    records = [] # Salva forecasts

    for i in dataset.index:
        origin_date = dataset.loc[i, "Date"]
        if not (origin_date >= oos_start and origin_date <= oos_end):
            continue

        future_index = i + h
        if future_index >= len(dataset):
            continue

        forecast_date = dataset.loc[future_index, "Date"]
        if forecast_date > oos_end:
            continue

        if pd.isna(dataset.loc[future_index, "VIX_log"]):
            continue

        # Creazione rolling window
        win_start = i - window_size
        win_end = i - 1

        if win_start < dataset.index.min():
            continue

        window = dataset.loc[win_start:win_end].copy()

        # Elimina eventuali NaN dal training set
        window = window.dropna(subset=features + [target])

        # Training set
        X_train = sm.add_constant(window[features], has_constant="add")
        y_train = window[target]

        # Check 
        if (X_train.isna().any().any() or y_train.isna().any()):
            continue

        # Stima OLS
        model = sm.OLS(y_train, X_train).fit()

        X_pred = sm.add_constant(dataset.loc[[i], features], has_constant="add")
        if X_pred.isna().any().any():
            continue

        # Forecast
        y_hat = float(model.predict(X_pred).iloc[0])
        y_true = float(dataset.loc[future_index, "VIX_log"])

        # Controllo NaN
        if not np.isfinite(y_hat):
            continue

        if not np.isfinite(y_true):
            continue

        # Esponenziale per convertire da log VIX a VIX
        VIX_true = np.exp(y_true)
        VIX_hat = np.exp(y_hat)
        VIX_origin = np.exp(float(dataset.loc[i, "VIX_log"]))


        # ----------------------
        # SALVATAGGIO RISULTATI
        # ----------------------
        records.append({
            "origin_date": origin_date,
            "forecast_date": forecast_date,
            "horizon": h,
            "y_true_log": y_true,
            "y_hat_log": y_hat,
            "VIX_origin": VIX_origin,
            "VIX_true": VIX_true,
            "VIX_hat": VIX_hat,
            "b0": model.params["const"],
            "bD": model.params["logVIX_d"],
            "bW": model.params["logVIX_w"],
            "bM": model.params["logVIX_m"],
            "R2_IS": model.rsquared
        })


    # ------------------------------
    # CREAZIONE DATAFRAME RISULTATI
    # ------------------------------
    results = pd.DataFrame(records)

    if results.empty:
        print("Nessuna previsione OOS generata")

        return results

    results = results.dropna(subset=["VIX_true", "VIX_hat"]).copy()


    # Controllo che il DataFrame non sia diventato vuoto
    if results.empty:
        print("Nessuna previsione valida dopo l'eliminazione dei NaN")

        return results


    # ---------------------
    # METRICHE DI FORECAST
    # ---------------------
    mse = mean_squared_error(results["VIX_true"], results["VIX_hat"])
    mae = mean_absolute_error(results["VIX_true"], results["VIX_hat"])
    mape = (mean_absolute_percentage_error(results["VIX_true"], results["VIX_hat"]) * 100)
    qlike = np.mean(results["VIX_true"] / results["VIX_hat"] - np.log(results["VIX_true"] / results["VIX_hat"]) - 1)

    # R^2
    denominator = np.sum((results["VIX_true"] - results["VIX_true"].mean()) ** 2)

    if denominator != 0:
        r2 = (1 - np.sum((results["VIX_true"] - results["VIX_hat"]) ** 2) / denominator)

    else:
        r2 = np.nan

    # Directional Accuracy
    results["actual_change"] = (results["VIX_true"] - results["VIX_origin"])
    results["predicted_change"] = (results["VIX_hat"] - results["VIX_origin"])
    results["actual_dir"] = np.sign(results["actual_change"])
    results["predicted_dir"] = np.sign(results["predicted_change"])

    valid = results[results["actual_dir"] != 0].copy()

    if len(valid) > 0:
        valid["hit"] = (valid["predicted_dir"] == valid["actual_dir"]).astype(int)
        directional_accuracy = (valid["hit"].mean())

    else:
        directional_accuracy = np.nan


    # ----------------
    # STAMPA METRICHE
    # ----------------
    print(f"MSE VIX: {mse:.6f}")
    print(f"MAE VIX: {mae:.6f}")
    print(f"MAPE VIX: {mape:.6f}%")
    print(f"QLIKE VIX: {qlike:.6f}")
    print(f"R^2 VIX: {r2:.6f}")
    print(f"Directional accuracy VIX: {directional_accuracy:.6f}")


    # ---------------
    # SALVA METRICHE
    # ---------------
    results["MSE"] = mse
    results["MAE"] = mae
    results["MAPE"] = mape
    results["QLIKE"] = qlike
    results["R2_OOS"] = r2
    results["Directional_Accuracy"] = (directional_accuracy)


    # --------
    # GRAFICO
    # --------
    plt.figure(figsize=(12, 5))
    plt.plot(
        results["forecast_date"],
        results["VIX_true"],
        label="VIX reale",
        color="black",
        linewidth=1.2
    )
    plt.plot(
        results["forecast_date"],
        results["VIX_hat"],
        label=f"VIX previsto HAR-type h={h}",
        color="orange",
        linewidth=1,
        linestyle="--"
    )
    plt.title(
        f"Confronto tra VIX reale e VIX previsto "
        f"(HAR-type, h={h})"
    )
    plt.xlabel("Data")
    plt.ylabel("VIX Level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


    # --------------
    # SALVA GRAFICO
    # --------------
    graph_path = (r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Econometrics\2. HAR-Type\Results\1_Grafici_backtest\HAR_VIX_h{h}.png")

    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------------
# CARTELLA PER SALVARE RISULTATI CSV
# -----------------------------------
output_dir = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\Repo\2_Econometrics\2_HAR-Type\Results"
os.makedirs(output_dir, exist_ok=True)


# ----------------------
# STIMA DEI TRE MODELLI
# ----------------------
results_h1 = run_har_vix(1)
results_h5 = run_har_vix(5)
results_h22 = run_har_vix(22)


# ----------------------
# SALVATAGGIO FORECASTS
# ----------------------
if not results_h1.empty:
    df_out_har_h1 = pd.DataFrame({
        "Actual": results_h1["VIX_true"].values,
        "Forecast": results_h1["VIX_hat"].values
    })

    df_out_har_h1.index = pd.to_datetime(results_h1["forecast_date"].values)
    df_out_har_h1.index.name = "Date"
    df_out_har_h1.to_csv(os.path.join(output_dir,"har_h1.csv"))

if not results_h5.empty:
    df_out_har_h5 = pd.DataFrame({
        "Actual": results_h5["VIX_true"].values,
        "Forecast": results_h5["VIX_hat"].values
    })

    df_out_har_h5.index = pd.to_datetime(results_h5["forecast_date"].values)
    df_out_har_h5.index.name = "Date"
    df_out_har_h5.to_csv(os.path.join(output_dir, "har_h5.csv"))

if not results_h22.empty:
    df_out_har_h22 = pd.DataFrame({
        "Actual": results_h22["VIX_true"].values,
        "Forecast": results_h22["VIX_hat"].values
    })

    df_out_har_h22.index = pd.to_datetime(results_h22["forecast_date"].values)
    df_out_har_h22.index.name = "Date"
    df_out_har_h22.to_csv(os.path.join(output_dir, "har_h22.csv"))


# --------------------
# RIEPILOGO RISULTATI
# --------------------
summary = pd.DataFrame({
    "Horizon": [1, 5, 22],
    "MSE": [
        results_h1["MSE"].iloc[0]
        if not results_h1.empty else np.nan,

        results_h5["MSE"].iloc[0]
        if not results_h5.empty else np.nan,

        results_h22["MSE"].iloc[0]
        if not results_h22.empty else np.nan
    ],

    "MAE": [
        results_h1["MAE"].iloc[0]
        if not results_h1.empty else np.nan,

        results_h5["MAE"].iloc[0]
        if not results_h5.empty else np.nan,

        results_h22["MAE"].iloc[0]
        if not results_h22.empty else np.nan
    ],

    "MAPE": [
        results_h1["MAPE"].iloc[0]
        if not results_h1.empty else np.nan,

        results_h5["MAPE"].iloc[0]
        if not results_h5.empty else np.nan,

        results_h22["MAPE"].iloc[0]
        if not results_h22.empty else np.nan
    ],

    "QLIKE": [
        results_h1["QLIKE"].iloc[0]
        if not results_h1.empty else np.nan,

        results_h5["QLIKE"].iloc[0]
        if not results_h5.empty else np.nan,

        results_h22["QLIKE"].iloc[0]
        if not results_h22.empty else np.nan
    ],

    "R2": [
        results_h1["R2_OOS"].iloc[0]
        if not results_h1.empty else np.nan,

        results_h5["R2_OOS"].iloc[0]
        if not results_h5.empty else np.nan,

        results_h22["R2_OOS"].iloc[0]
        if not results_h22.empty else np.nan
    ],

    "Directional_Accuracy": [
        results_h1["Directional_Accuracy"].iloc[0]
        if not results_h1.empty else np.nan,

        results_h5["Directional_Accuracy"].iloc[0]
        if not results_h5.empty else np.nan,

        results_h22["Directional_Accuracy"].iloc[0]
        if not results_h22.empty else np.nan
    ]

})

print(summary.to_string(index=False))

summary.to_csv(os.path.join(output_dir, "har_summary.csv"), index=False)


