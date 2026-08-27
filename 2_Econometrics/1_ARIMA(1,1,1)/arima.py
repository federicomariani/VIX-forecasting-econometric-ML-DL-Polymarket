import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")



# ---------
# PERCORSI 
# ---------
base_path = Path(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\Repo\2_Econometrics\1_ARIMA(1,1,1)")

path_input = base_path / "dataset_econometrics.csv"
dir_results = base_path / "Results" 
dir_plots = base_path / "Results" 

dir_results.mkdir(parents=True, exist_ok=True)
dir_plots.mkdir(parents=True, exist_ok=True)


# -------------------------
# IMPORT DATASET E PULIZIA
# -------------------------
df = pd.read_csv(path_input)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Date"]).drop_duplicates(subset="Date", keep="first")
df = df.sort_values("Date").reset_index(drop=True)

in_sample_start = pd.Timestamp("2014-01-01")
in_sample_end = pd.Timestamp("2021-12-31")

values = df["VIX_log"].values
dates = df["Date"].values
n = len(values)

# Indice della prima data target OOS 
target_start_idx = int(df.index[df["Date"] >= pd.Timestamp("2022-01-03")][0])

# Calcolo ampiezza finestra rolling fissa
train_start_idx = int(df.index[df["Date"] >= in_sample_start][0])
train_end_idx = int(df.index[df["Date"] <= in_sample_end][-1])
window_size = train_end_idx - train_start_idx + 1

horizons = [1, 5, 22]


# -----------------
# CALCOLO METRICHE
# -----------------
def calc_metrics(actual, forecast, origin_actual):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    o = np.asarray(origin_actual, dtype=float)

    mask = ~np.isnan(a) & ~np.isnan(f)
    a, f, o = a[mask], f[mask], o[mask]

    mae = np.mean(np.abs(a - f))
    mse = np.mean((a - f) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((a - f) / a)) * 100
    qlike = np.mean(a / f - np.log(a / f) - 1)

    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pred_dir = np.sign(f - o)
    real_dir = np.sign(a - o)
    valid_dir = real_dir != 0
    da = np.mean(pred_dir[valid_dir] == real_dir[valid_dir]) * 100 if valid_dir.sum() > 0 else np.nan

    return {
        "n": mask.sum(), "MAE": mae, "RMSE": rmse, "MSE": mse,
        "MAPE": mape, "QLIKE": qlike, "R2": r2, "Directional_Accuracy": da
    }


# ------------------------
# ROLLING WINDOW BACKTEST
# ------------------------
metrics_summary = {}

for h in horizons:
    print(f"ELABORAZIONE ORIZZONTE h = {h}")
    
    start_origin_idx = target_start_idx - h
    
    records = []
    t0 = time.time()
    total_steps = n - h - start_origin_idx

    for step_count, t_origin in enumerate(range(start_origin_idx, n - h)):
        # Finestra rolling mobile a dimensione fissa
        window_data = values[t_origin - window_size + 1 : t_origin + 1]

        try:
            model = ARIMA(window_data, order=(1, 1, 1))
            fitted = model.fit()
            fc_val = np.exp(fitted.forecast(steps=h)[-1])
        except Exception:
            fc_val = np.nan

        target_idx = t_origin + h

        records.append({
            "target_date": pd.Timestamp(dates[target_idx]),
            "origin_date": pd.Timestamp(dates[t_origin]),
            "origin_actual": np.exp(values[t_origin]),
            "actual": np.exp(values[target_idx]),
            "forecast": fc_val
        })

        if (step_count + 1) % 100 == 0 or (step_count + 1) == total_steps:
            elapsed = time.time() - t0
            print(f"  [h={h}] Passi completati: {step_count + 1}/{total_steps} ({elapsed:.1f}s)")

    df_h_results = pd.DataFrame(records)

    # CSV
    df_mcs_export = (
        df_h_results
        .set_index("target_date")[["actual", "forecast"]]
        .rename(columns={"actual": "Actual", "forecast": "Forecast"})
    )
    df_mcs_export.index = df_mcs_export.index.strftime("%d/%m/%Y")
    df_mcs_export.index.name = "Date"

    path_csv_mcs = dir_results / f"arima_forecast_h{h}_aligned.csv"
    df_mcs_export.to_csv(path_csv_mcs)
    
    # Calcolo metriche per ogni orizzonte
    metrics_summary[h] = calc_metrics(
        df_h_results["actual"],
        df_h_results["forecast"],
        df_h_results["origin_actual"]
    )

    # Grafici
    plot_df = df_h_results.set_index("target_date")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df.index, plot_df["actual"], label="VIX Reale", color="black", linewidth=1.8, alpha=0.4)
    ax.plot(plot_df.index, plot_df["forecast"], label=f"Previsione ARIMA h={h}",
            color="tab:blue", linestyle="--", linewidth=1.2)

    ax.set_title(f"VIX Reale vs Previsione ARIMA(1,1,1) - Orizzonte h={h}")
    ax.set_xlabel("Data Target")
    ax.set_ylabel("VIX")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path_plot = dir_plots / f"arima_forecast_h{h}.png"
    plt.savefig(path_plot, dpi=150)
    plt.close()


# ----------------------------
# SALVATAGGIO METRICHE FINALI
# ----------------------------
metrics_df = pd.DataFrame(metrics_summary).T
metrics_df.index.name = "horizon"

print(metrics_df.to_string())