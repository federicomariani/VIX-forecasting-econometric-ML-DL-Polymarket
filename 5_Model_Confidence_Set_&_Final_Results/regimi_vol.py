import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from pathlib import Path


# ---------------
# IMPORT DATASET
# ---------------
# Cambiare cartella ogni volta che si runna un orizzonte (h) diverso
folder = Path(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\5. Model Confidence Set\+ Polymarkets\0_Forecast\h22")


# -----------------
# CALCOLO METRICHE
# -----------------
results = []

for file in folder.glob("*.csv"):

    # Import dataset
    dataset = pd.read_csv(file).dropna().copy()

    # Assicurarsi che le date siano ordinate
    dataset["Date"] = pd.to_datetime(dataset["Date"], dayfirst=True)
    dataset = dataset.sort_values("Date").reset_index(drop=True)    

    # Regimi di volatilità 
    dataset["Volatility Regime"] = np.select(
    [
        dataset["Actual"] < 20,
        (dataset["Actual"] >= 20) & (dataset["Actual"] <= 30),
        dataset["Actual"] > 30
    ],
    [
        "Low (<20)",
        "Medium (20-30)",
        "High (>30)"
    ],
    default="Unknown"
    )

    # Calcolo metriche errore per ogni regime di volatilità
    for regime in ["Low (<20)", "Medium (20-30)", "High (>30)"]:

        # Filtra le osservazioni del regime
        regime_data = dataset[dataset["Volatility Regime"] == regime].copy()

        # Se non ci sono osservazioni, passa al regime successivo
        if len(regime_data) == 0:
            continue

        y_true = regime_data["Actual"]
        y_predicted = regime_data["Forecast"]


        mse = mean_squared_error(y_true, y_predicted)

        mae = mean_absolute_error(y_true, y_predicted)

        mape = mean_absolute_percentage_error(y_true, y_predicted)

        r2 = r2_score(y_true, y_predicted)

        qlike = np.mean((y_true / y_predicted) - np.log(y_true / y_predicted)- 1)

        # Directional Accuracy
        actual_direction = np.sign(dataset["Actual"].diff())

        predicted_direction = np.sign(dataset["Forecast"] - dataset["Actual"].shift(1))

        # Seleziona solo le osservazioni appartenenti al regime corrente
        regime_mask = (dataset["Volatility Regime"] == regime)

        directional_accuracy = (actual_direction[regime_mask] == predicted_direction[regime_mask]).mean() * 100

        # Salva risultati    
        results.append({
            "Model": file.stem,
            "Volatility Regime": regime,
            "N": len(regime_data),
            "MSE": mse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "QLIKE": qlike,
            "Directional Accuracy": directional_accuracy
        })


# ----------------------------
# CREAZIONE TABELLA RISULTATI
# ----------------------------
results_df = pd.DataFrame(results)


# Ordine desiderato dei regimi
regime_order = [
    "Low (<20)",
    "Medium (20-30)",
    "High (>30)"
]

results_df["Volatility Regime"] = pd.Categorical(
    results_df["Volatility Regime"],
    categories=regime_order,
    ordered=True
)

results_df = results_df.sort_values(["Model", "Volatility Regime"])
vol_regime_h1 = results_df
vol_regime_h1.to_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\5. Model Confidence Set\+ Polymarkets\1_Results\vol_regime_h22.csv", index=False)

print(vol_regime_h1.to_string(index=False))