import pandas as pd
import numpy as np
from pathlib import Path
from dieboldmariano import dm_test



# --------------
# CONFIGURATION
# --------------
orizzonte = 22 # change manually each time the code is run

base_path = Path(r"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\With_Polymarket")

model_folders = {
    "econom": base_path / "Modelli_econometrici" / f"h{orizzonte}",
    "ml": base_path / "Machine_Learning" / f"h{orizzonte}",
    "dl": base_path / "Deep_Learning" / f"h{orizzonte}",
}

# Explicit definition of the 1-vs-1 PAIRS (with PM vs without PM)
pairs = [
    ("ARIMA", model_folders["econom"], f"arima_h{orizzonte}_con_PM.csv", f"arima_h{orizzonte}_senza_PM.csv"),
    ("HAR", model_folders["econom"], f"har_h{orizzonte}_con_PM.csv", f"har_h{orizzonte}_senza_PM.csv"),
    ("Random Forest CV", model_folders["ml"], f"random_forest_gridsearch_h{orizzonte}_con_PM.csv", f"random_forest_gridsearch_h{orizzonte}_senza_PM.csv"),
    ("Random Forest BO", model_folders["ml"], f"random_forest_bayesoptimization_h{orizzonte}_con_PM.csv", f"random_forest_bayesoptimization_h{orizzonte}_senza_PM.csv"),
    ("XGBoost CV", model_folders["ml"], f"xgboost_gridsearch_h{orizzonte}_con_PM.csv", f"xgboost_gridsearch_h{orizzonte}_senza_PM.csv"),
    ("XGBoost BO", model_folders["ml"], f"xgboost_bayesoptimization_h{orizzonte}_con_PM.csv", f"xgboost_bayesoptimization_h{orizzonte}_senza_PM.csv"),
    ("LSTM CV", model_folders["dl"], f"lstm_gridsearch_h{orizzonte}_con_PM.csv", f"lstm_gridsearch_h{orizzonte}_senza_PM.csv"),
    ("LSTM BO", model_folders["dl"], f"lstm_bayesoptimization_h{orizzonte}_con_PM.csv", f"lstm_bayesoptimization_h{orizzonte}_senza_PM.csv"),
    ("TFT CV", model_folders["dl"], f"tft_gridsearch_h{orizzonte}_con_PM.csv", f"tft_gridsearch_h{orizzonte}_senza_PM.csv"),
    ("TFT BO", model_folders["dl"], f"tft_bayesoptimization_h{orizzonte}_con_PM.csv", f"tft_bayesoptimization_h{orizzonte}_senza_PM.csv"),
]


# -----------------------------
# LOSS FUNCTIONS (MSE & QLIKE)
# -----------------------------
def compute_mse_loss(actual, forecast):
    return (actual - forecast) ** 2

def compute_qlike_loss(actual, forecast):
    return (actual / forecast) - np.log(actual / forecast) - 1


# -------------------------
# RUN DIEBOLD-MARIANO TEST 
# -------------------------
results_mse = []
results_qlike = []

for model_label, folder, file_con, file_senza in pairs:
    # Load the "with PM" and "without PM" files
    df_con = pd.read_csv(folder / file_con).dropna().set_index("Date")
    df_senza = pd.read_csv(folder / file_senza).dropna().set_index("Date")
    
    # Align by date
    common_idx = df_con.index.intersection(df_senza.index)
    df_con = df_con.loc[common_idx]
    df_senza = df_senza.loc[common_idx]
    
    act = df_con["Actual"].values
    pred_con = df_con["Forecast"].values
    pred_senza = df_senza["Forecast"].values
    
    # -------------
    # RUN MSE TEST
    # -------------
    # Use variance_estimator='bartlett' to avoid negative variances at h=22
    try:
        stat_mse, p_val_mse = dm_test(act, pred_con, pred_senza, h=orizzonte, variance_estimator='bartlett')
    except Exception as e:
        stat_mse, p_val_mse = np.nan, np.nan

    loss_mse_con = compute_mse_loss(act, pred_con)
    loss_mse_senza = compute_mse_loss(act, pred_senza)
    mean_mse_con = np.mean(loss_mse_con)
    mean_mse_senza = np.mean(loss_mse_senza)
    
    results_mse.append({
        "Model": model_label,
        "MSE (with PM)": mean_mse_con,
        "MSE (without PM)": mean_mse_senza,
        "Diff Loss (%)": ((mean_mse_con - mean_mse_senza) / mean_mse_senza) * 100,
        "Stat DM": stat_mse,
        "p-value": p_val_mse,
        "Significant (alpha=0.05)": "Sì ***" if (pd.notna(p_val_mse) and p_val_mse < 0.05) else "No"
    })
    
    # ---------------
    # RUN QLIKE TEST
    # ---------------
    loss_qlike_con = compute_qlike_loss(act, pred_con)
    loss_qlike_senza = compute_qlike_loss(act, pred_senza)
    zeros = np.zeros_like(act)
    
    try:
        stat_qlike, p_val_qlike = dm_test(zeros, loss_qlike_con, loss_qlike_senza, h=orizzonte, variance_estimator='bartlett')
    except Exception as e:
        stat_qlike, p_val_qlike = np.nan, np.nan
        
    mean_qlike_con = np.mean(loss_qlike_con)
    mean_qlike_senza = np.mean(loss_qlike_senza)
    
    results_qlike.append({
        "Model": model_label,
        "QLIKE (with PM)": mean_qlike_con,
        "QLIKE (without PM)": mean_qlike_senza,
        "Diff Loss (%)": ((mean_qlike_con - mean_qlike_senza) / mean_qlike_senza) * 100,
        "Stat DM": stat_qlike,
        "p-value": p_val_qlike,
        "Significant (alpha=0.05)": "Sì ***" if (pd.notna(p_val_qlike) and p_val_qlike < 0.05) else "No"
    })

# --------------
# PRINT RESULTS
# --------------
print(f"\n==================================================================")
print(f" DIEBOLD-MARIANO TEST (library 'dieboldmariano') - Forecast horizon h = {orizzonte}")
print(f" 1-vs-1 comparison: Same model (with PM vs without PM)")
print(f"==================================================================")

print("\n--- MSE RESULTS ---")
print(pd.DataFrame(results_mse).to_string(index=False))

print("\n--- QLIKE RESULTS ---")
print(pd.DataFrame(results_qlike).to_string(index=False))