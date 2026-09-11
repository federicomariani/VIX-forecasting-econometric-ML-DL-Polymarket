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
base_path = Path(r"C:\Users\fede1\Desktop\Repo\0_Dataset\Polymarket_data")

path_input = base_path / "dataset_w_polymarket.csv"
dir_results = Path(r"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket\Modelli_econometrici")
dir_plots = dir_results 

dir_results.mkdir(parents=True, exist_ok=True)
dir_plots.mkdir(parents=True, exist_ok=True)


# -------------------------------
# CON O SENZA FEATURE POLYMARKET
# -------------------------------
evento = "con_PM"   # "con_PM" oppure "senza_PM"

pm_features = [
    "cut_50", "cut_25", "hold", "hike_25",
    "expected_rate_change_fomc", "rate_uncertainty_fomc",
    "probability_cut_fomc", "probability_hike_fomc", "entropy_fomc",
    "us_recession", "entropy_us_recession",
    "change_1d_us_recession", "change_5d_us_recession",
    "cpi_max_2_7", "cpi_2_8", "cpi_2_9", "cpi_min_3_0",
    "expected_cpi", "uncertainty_cpi", "entropy_cpi",
]

exog_cols = pm_features if evento == "con_PM" else []


# -------------------------
# IMPORT DATASET E PULIZIA
# -------------------------
df = pd.read_csv(
    path_input,
    index_col="Date",
    parse_dates=["Date"],
    dayfirst=True,
)
df.index.name = "Date"
df = df.sort_index()

horizons = [1, 5, 22]
orizzonte = 1  # variabile usata solo per stampare splitting dati
df = df.iloc[:-orizzonte].copy()

subset_cols = ["Close_VIX"] + exog_cols
df = df.dropna(subset=subset_cols).copy()

values = np.log(df["Close_VIX"]).values
dates = df.index.values
n = len(values)

# DataFrame delle feature Polymarket "grezze" (non laggate): serve sia per
# costruire la versione laggata sia per il forecast (vedi sotto)
exog_raw = df[exog_cols].reset_index(drop=True) if exog_cols else None

# Suddivisione cronologica: 70% training e 30% test
train_end_idx = int(n * 0.70)
target_start_idx = train_end_idx
window_size = train_end_idx

train_data = df.iloc[:train_end_idx]
test_data = df.iloc[target_start_idx:]

print("\nSUDDIVISIONE DATASET")
print(
    f"Training set: {len(train_data)} osservazioni "
    f"({len(train_data) / n:.2%}) | "
    f"{train_data.index[0].date()} - {train_data.index[-1].date()}"
)
print(
    f"Test set: {len(test_data)} osservazioni "
    f"({len(test_data) / n:.2%}) | "
    f"{test_data.index[0].date()} - {test_data.index[-1].date()}"
)


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


# ------------------------------------------------------------------
# COSTRUZIONE FEATURE POLYMARKET LAGGATE (lag = h, per ogni orizzonte)
# ------------------------------------------------------------------
# PM_lag_t = PM_(t-h)
#
# Perche' questo lag evita il problema della persistenza in ARIMAX:
# per il forecast a h passi serve un valore di exog per ciascuno dei
# giorni futuri T+1, ..., T+h. Con la feature laggata di h periodi,
# il valore richiesto in T+k e' PM_(T+k-h): per k che va da 1 a h,
# questo indice va da T+1-h a T, cioe' SEMPRE minore o uguale a T
# (oggi). Non serve quindi alcuna assunzione sui valori futuri di
# Polymarket: tutti i valori necessari sono gia' osservati.
def build_lagged_exog(exog_raw_df, h):
    if exog_raw_df is None:
        return None
    lagged = exog_raw_df.shift(h)
    lagged.columns = [f"{c}_lag{h}" for c in exog_raw_df.columns]
    return lagged


# ------------------------
# ROLLING WINDOW BACKTEST
# ------------------------
metrics_summary = {}

for h in horizons:
    print(f"\nELABORAZIONE ORIZZONTE h = {h}")

    exog_lagged_h = build_lagged_exog(exog_raw, h) if exog_cols else None

    # Si sposta l'inizio del test di (h-1) osservazioni per garantire che la
    # prima finestra rolling non contenga NaN dovuti al lag (le prime h
    # osservazioni di exog_lagged_h sono NaN per costruzione, essendo shift(h))
    start_origin_idx = target_start_idx + (h - 1) if exog_cols else target_start_idx

    records = []
    t0 = time.time()
    total_steps = n - h - start_origin_idx

    for step_count, t_origin in enumerate(range(start_origin_idx, n - h)):
        window_data = values[t_origin - window_size + 1: t_origin + 1]

        if exog_cols:
            exog_window = exog_lagged_h.iloc[t_origin - window_size + 1: t_origin + 1].reset_index(drop=True)
            # exog futuro per il forecast a h passi: valori GIA' OSSERVATI
            # (vedi spiegazione sopra: PM_raw da T-h+1 a T)
            exog_future = exog_raw.iloc[t_origin - h + 1: t_origin + 1].reset_index(drop=True)
            exog_future.columns = exog_window.columns
        else:
            exog_window = None
            exog_future = None

        try:
            model = ARIMA(window_data, order=(1, 1, 1), exog=exog_window)
            fitted = model.fit()
            fc_raw = fitted.forecast(steps=h, exog=exog_future)
            fc_val = np.exp(np.asarray(fc_raw)[-1])
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

    path_csv_mcs = dir_results / f"arima_h{h}_{evento}.csv"
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
    ax.plot(plot_df.index, plot_df["forecast"], label=f"Previsione ARIMAX h={h} ({evento})",
            color="tab:blue", linestyle="--", linewidth=1.2)

    ax.set_title(f"VIX Reale vs Previsione ARIMAX(1,1,1) con PM laggate - Orizzonte h={h} ({evento})")
    ax.set_xlabel("Data Target")
    ax.set_ylabel("VIX")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path_plot = dir_plots / f"arima_h{h}_{evento}.png"
    plt.savefig(path_plot, dpi=150)
    plt.close()


# ----------------------------
# SALVATAGGIO METRICHE FINALI
# ----------------------------
metrics_df = pd.DataFrame(metrics_summary).T
metrics_df.index.name = "horizon"

print("\n" + metrics_df.to_string())

metrics_df.to_csv(dir_results / f"arima_{evento}.csv")