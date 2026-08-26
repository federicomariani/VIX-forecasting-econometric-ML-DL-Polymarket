import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from pathlib import Path



# ---------------
# IMPORT DATASET
# ---------------
# Cambiare cartella ogni volta che si runna un orizzonte (h) diverso
folder = Path(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\5. Model Confidence Set\0_Forecast\h22")


# -----------------
# CALCOLO METRICHE
# -----------------
results = []

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