import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from pathlib import Path



# ---------------
# IMPORT DATASET
# ---------------
# Cambiare cartella ogni volta che si runna un orizzonte (h) diverso
orizzonte = 22 # Da cambiare ad ogni run (1, 5, 22)

folders = [
    Path(rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Modelli_econometrici\h{orizzonte}"),
    Path(rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Machine_Learning\h{orizzonte}"),
    Path(rf"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Deep_Learning\h{orizzonte}")
]



# -----------------
# CALCOLO METRICHE
# -----------------
results = []

for folder in folders:
    for file in folder.glob("*.csv"):
        dataset = pd.read_csv(file).dropna()

        y_true = dataset["Actual"]
        y_predicted = dataset["Forecast"]

        mse = mean_squared_error(y_true, y_predicted)
        mae = mean_absolute_error(y_true, y_predicted)
        mape = mean_absolute_percentage_error(y_true, y_predicted)
        r2 = r2_score(y_true, y_predicted)

        qlike = np.mean((y_true / y_predicted) - np.log(y_true / y_predicted) - 1)

        actual_direction = np.sign(y_true.diff())
        predicted_direction = np.sign(y_predicted - y_true.shift(1))

        directional_accuracy = (actual_direction == predicted_direction).iloc[1:].mean() * 100

        results.append({
            "Model": file.stem,
            "MSE": mse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "QLIKE": qlike,
            "Directional Accuracy": directional_accuracy
        })


results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))