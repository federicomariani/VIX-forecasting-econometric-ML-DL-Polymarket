import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from pathlib import Path


# ---------------
# IMPORT DATASET
# ---------------
orizzonte = 22 # Da cambiare ad ogni run (1, 5 e 22)

project_root = Path(__file__).resolve().parent.parent
forecasts_root = project_root / "5_Forecasts_&_Error_Metrics" / "Normale"

folders = [
    forecasts_root / "Modelli_econometrici" / f"h{orizzonte}",
    forecasts_root / "Machine_Learning" / f"h{orizzonte}",
    forecasts_root / "Deep_Learning" / f"h{orizzonte}",
]

source_dataset = pd.read_csv(
    project_root / "0_Dataset" / "Data" / "Clean" / "dataset.csv",
    parse_dates=["Date"],
).sort_values("Date").reset_index(drop=True)

if source_dataset["Date"].duplicated().any():
    raise ValueError("Il dataset sorgente contiene date duplicate.")

forecast_files = []
for folder in folders:
    if not folder.is_dir():
        raise FileNotFoundError(f"Cartella dei forecast non trovata: {folder}")

    for file in folder.glob("*.csv"):
        forecast_files.append(file)

forecast_datasets = {}
forecast_date_sets = []
forecast_actuals = []
for file in forecast_files:
    dataset = pd.read_csv(file).dropna().copy()
    dataset["Date"] = pd.to_datetime(
        dataset["Date"],
        format="mixed",
        dayfirst=True,
    )
    dataset = dataset.sort_values("Date").reset_index(drop=True)
    if dataset["Date"].duplicated().any():
        raise ValueError(f"Date duplicate nel dataset dei forecast: {file.name}")

    forecast_datasets[file] = dataset
    forecast_date_sets.append(set(dataset["Date"]))
    forecast_actuals.append(dataset[["Date", "Actual"]])

if not forecast_date_sets:
    raise ValueError(f"Nessun file forecast trovato per h={orizzonte}.")

common_dates = set.intersection(*forecast_date_sets)
if not common_dates:
    raise ValueError(f"Nessuna data forecast comune tra i modelli per h={orizzonte}.")

if forecast_actuals:
    forecast_actuals = pd.concat(forecast_actuals, ignore_index=True)
    actual_ranges = forecast_actuals.groupby("Date")["Actual"].agg(["min", "max"])
    conflicting_dates = actual_ranges.index[(actual_ranges["max"] - actual_ranges["min"]).abs() > 1e-5]
    if len(conflicting_dates):
        raise ValueError(
            "Valori Actual discordanti tra i forecast per le date: "
            f"{conflicting_dates.strftime('%Y-%m-%d').tolist()[:5]}"
        )
    forecast_actuals = forecast_actuals.drop_duplicates("Date").set_index("Date")["Actual"]
else:
    forecast_actuals = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="Date"))

source_vix = source_dataset.set_index("Date")["Close_VIX"]
# Ripristina le date eliminate dal dataset pulito per valori mancanti in altre variabili.
vix_series = source_vix.combine_first(forecast_actuals).sort_index()
source_dates = pd.DatetimeIndex(vix_series.index)
source_vix = vix_series.rename("Close_VIX")


# -----------------
# CALCOLO METRICHE
# -----------------
results = []

for file in forecast_files:

    # Valuta tutti i modelli sulle stesse date per confrontare le metriche.
    dataset = forecast_datasets[file]
    dataset = dataset[dataset["Date"].isin(common_dates)].copy().reset_index(drop=True)

    target_positions = source_dates.get_indexer(dataset["Date"])
    origin_positions = target_positions - orizzonte
    invalid_positions = (target_positions < 0) | (origin_positions < 0)
    if invalid_positions.any():
        invalid_dates = dataset.loc[invalid_positions, "Date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"Date target non presenti nel dataset sorgente o senza osservazione "
            f"d'origine per h={orizzonte} in {file.name}: {invalid_dates[:5]}"
        )

    # Associa ciascun forecast alla data d'origine e al Close_VIX osservato in quella data.
    dataset["Origin Date"] = source_dates[origin_positions]
    dataset = dataset.join(source_vix.rename("Close_VIX"), on="Origin Date")
    if dataset["Close_VIX"].isna().any():
        raise ValueError(f"Close_VIX mancante alla data d'origine in {file.name}.")

    # Regimi di volatilità definiti sul VIX all'origine
    dataset["Volatility Regime"] = np.select(
        [
        dataset["Close_VIX"] < 20,
            (dataset["Close_VIX"] >= 20) & (dataset["Close_VIX"] <= 30),
            dataset["Close_VIX"] > 30
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

        # Direzione reale e prevista rispetto al VIX osservato all'origine del forecast.
        actual_direction = np.sign(dataset["Actual"] - dataset["Close_VIX"])
        predicted_direction = np.sign(dataset["Forecast"] - dataset["Close_VIX"])
        regime_mask = dataset["Volatility Regime"] == regime
        directional_accuracy = (
            actual_direction[regime_mask] == predicted_direction[regime_mask]
        ).mean() * 100

        # Salva metriche aggregate per modello e regime.
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

    # Conserva anche il dettaglio giornaliero: il Close_VIX resta associato alla sua data d'origine.
    detail = dataset[["Date", "Origin Date", "Close_VIX", "Actual", "Forecast", "Volatility Regime"]].copy()
    detail.insert(0, "Model", file.stem)
    detailed_results = detail if "detailed_results" not in locals() else pd.concat(
        [detailed_results, detail], ignore_index=True
    )


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
vol_regime = results_df

output_folder = project_root / "6_Final_Results" / "Results"
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / f"vol_regime_h{orizzonte}.csv"
vol_regime.to_csv(output_path, index=False)

detailed_results = detailed_results.sort_values(["Model", "Date"])
detailed_output_path = output_folder / f"vol_regime_detail_h{orizzonte}.csv"
detailed_results.to_csv(detailed_output_path, index=False)

print(vol_regime.to_string(index=False))
