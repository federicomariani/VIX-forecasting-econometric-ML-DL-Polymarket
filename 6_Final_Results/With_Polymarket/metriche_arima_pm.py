import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


# Orizzonte da valutare: 1, 5 oppure 22 giorni.
orizzonte = 22
modello = "har" # har o arima

project_root = Path(__file__).resolve().parents[2]
forecast_folder = (
    project_root
    / "5_Forecasts_&_Error_Metrics"
    / "With_Polymarket"
    / "Modelli_econometrici"
    / f"h{orizzonte}"
)

files = {
    "con_PM": forecast_folder / f"{modello}_h{orizzonte}_con_PM.csv",
    "senza_PM": forecast_folder / f"{modello}_h{orizzonte}_senza_PM.csv",
}


def load_forecasts(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File forecast non trovato: {path}")

    data = pd.read_csv(path)
    required = {"Date", "Actual", "Forecast"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {path.name}: {sorted(missing)}")

    data = data.dropna(subset=["Date", "Actual", "Forecast"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], format="mixed", dayfirst=True)
    if data["Date"].duplicated().any():
        raise ValueError(f"Date duplicate nel file {path.name}")
    return data.set_index("Date").sort_index()[["Actual", "Forecast"]]


# Le due versioni vengono valutate sulle sole date target presenti in entrambi i file.
forecasts = {label: load_forecasts(path) for label, path in files.items()}
common_dates = forecasts["con_PM"].index.intersection(forecasts["senza_PM"].index)
if common_dates.empty:
    raise ValueError(f"Non ci sono date comuni tra i forecast {modello} con e senza PM.")

aligned = {
    label: data.loc[common_dates].sort_index()
    for label, data in forecasts.items()
}
actual_con = aligned["con_PM"]["Actual"]
actual_senza = aligned["senza_PM"]["Actual"]
if not np.allclose(actual_con, actual_senza, rtol=0, atol=1e-5):
    raise ValueError("I valori Actual differiscono tra i due forecast sulle date comuni.")


def calculate_metrics(data: pd.DataFrame) -> dict:
    actual = data["Actual"]
    forecast = data["Forecast"]

    directional_actual = np.sign(actual.diff())
    directional_forecast = np.sign(forecast - actual.shift(1))
    directional_accuracy = (
        directional_actual.eq(directional_forecast).iloc[1:].mean() * 100
    )

    return {
        "N": len(data),
        "MSE": mean_squared_error(actual, forecast),
        "MAE": mean_absolute_error(actual, forecast),
        "MAPE": mean_absolute_percentage_error(actual, forecast),
        "R2": r2_score(actual, forecast),
        "QLIKE": np.mean((actual / forecast) - np.log(actual / forecast) - 1),
        "Directional Accuracy": directional_accuracy,
        "Prima data": data.index.min().strftime("%d/%m/%Y"),
        "Ultima data": data.index.max().strftime("%d/%m/%Y"),
    }


output_folder = (
    project_root
    / "5_Forecasts_&_Error_Metrics"
    / "With_Polymarket"
    / "Modelli_econometrici"
)
output_folder.mkdir(parents=True, exist_ok=True)

for label, data in aligned.items():
    result = pd.DataFrame([calculate_metrics(data)])
    output_path = output_folder / f"metriche_{modello}_h{orizzonte}_{label}.csv"
    result.to_csv(output_path, index=False)
    print(f"\n{modello} {label} (N={len(data)}, date comuni)")
    print(result.to_string(index=False))
    print(f"Salvato: {output_path}")
