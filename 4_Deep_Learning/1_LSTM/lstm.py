import os
import time
import random
import numpy as np
import pandas as pd
np.int = int  # serve per compatibilità con skopt

import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from skopt import BayesSearchCV
from skopt.space import Categorical


# --------------
# SEED E DEVICE
# --------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = "cpu"


# ---------------
# IMPORT DATASET
# ---------------
dataset = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\3. Deep Learning\0. Dataset\Data\Clean\dataset.csv", index_col=["Date"])


# -----------------
# DATA PREPARATION
# -----------------
# Orizzonte di previsione: 1, 5 e 22 giorni 
orizzonte = 1 # da cambiare manualmente ad ogni run del codice

lookback = 20  # lunghezza della finestra di input 

# Creazione target shiftato in avanti di h giorni (previsione diretta multi-step)
target_col = f"Close_VIX_h{orizzonte}"
dataset[target_col] = dataset["Close_VIX"].shift(-orizzonte)

# Rimuovo le ultime righe che non hanno un target valido (fine serie)
dataset_h = dataset.dropna(subset=[target_col])

# Predittori completi (17 variabili, come nel modello Random Forest)
predittori = ["Log_Return_S&P500", "Log_Return_WTI", "Close_OVX", "OVX_VIX_spread",
              "VIX_lag1", "VIX_lag5", "VIX_lag22", "RSI_14d", "Log_Return_DXY",
              "Yield_curve", "Log_difference_initial_claims", "VIX_MA5", "VIX_MA10", 
              "VIX_MA20", "Credit_Spread", "Log_Return_Gold", "Weekday"]
target = [target_col]


# ------------------
# DATA MANIPULATION
# ------------------
# Split cronologico (stesse date del modello Random Forest)
X_training = dataset_h.loc[:"2019-12-31", predittori]
y_training = dataset_h.loc[:"2019-12-31", target].squeeze()

X_validation = dataset_h.loc["2020-01-01":"2021-12-31", predittori]
y_validation = dataset_h.loc["2020-01-01":"2021-12-31", target].squeeze()

X_test = dataset_h.loc["2022-01-01":, predittori]
y_test = dataset_h.loc["2022-01-01":, target].squeeze()

X_train_validation = pd.concat([X_training, X_validation])
y_train_validation = pd.concat([y_training, y_validation])


# ------------------------------
# DEFINIZIONE DEL MODELLO LSTM 
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=10, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # output dell'ultimo timestep
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# Parametri fissi condivisi da tutte le istanze del wrapper skorch
kwargs_comuni = dict(
    optimizer=torch.optim.Adam,
    criterion=nn.MSELoss,
    train_split=None,   # la validazione è già gestita esternamente (PredefinedSplit / backtest)
    iterator_train__shuffle=False,  # ordine cronologico preservato
    verbose=0,
    device=device,
)


# ---------------------------------------------------
# FEATURE SELECTION (Permutation Importance su LSTM)
# ---------------------------------------------------
# Scaling preliminare, solo per questa fase esplorativa (fit su training)
scaler_X_fs = RobustScaler()
scaler_y_fs = RobustScaler()
scaler_X_fs.fit(X_training)
scaler_y_fs.fit(y_training.values.reshape(-1, 1))

X_training_scaled_fs = scaler_X_fs.transform(X_training)
y_training_scaled_fs = scaler_y_fs.transform(y_training.values.reshape(-1, 1)).ravel()
X_validation_scaled_fs = scaler_X_fs.transform(X_validation)
y_validation_scaled_fs = scaler_y_fs.transform(y_validation.values.reshape(-1, 1)).ravel()

# Tensorizzazione training (sequenze 3D)
X_training_seq_fs, y_training_seq_fs = [], []
for t in range(lookback, len(X_training_scaled_fs)):
    X_training_seq_fs.append(X_training_scaled_fs[t - lookback:t, :])
    y_training_seq_fs.append(y_training_scaled_fs[t])
X_training_seq_fs = np.array(X_training_seq_fs, dtype=np.float32)
y_training_seq_fs = np.array(y_training_seq_fs, dtype=np.float32).reshape(-1, 1)

# Tensorizzazione validation (sequenze 3D)
X_validation_seq_fs, y_validation_seq_fs = [], []
for t in range(lookback, len(X_validation_scaled_fs)):
    X_validation_seq_fs.append(X_validation_scaled_fs[t - lookback:t, :])
    y_validation_seq_fs.append(y_validation_scaled_fs[t])
X_validation_seq_fs = np.array(X_validation_seq_fs, dtype=np.float32)
y_validation_seq_fs = np.array(y_validation_seq_fs, dtype=np.float32).reshape(-1, 1)

# Modello LSTM baseline (iperparametri di default, non ottimizzati)
lstm_baseline = NeuralNetRegressor(
    module=LSTMModel,
    module__input_size=len(predittori),
    module__hidden_size=10,
    module__num_layers=2,
    module__dropout=0.3,
    optimizer__lr=0.001,
    batch_size=32,
    max_epochs=50,
    **kwargs_comuni,
)
lstm_baseline.fit(X_training_seq_fs, y_training_seq_fs)

# Permutation importance calcolata su validation (shuffling coerente su tutti i timestep)
y_pred_baseline = lstm_baseline.predict(X_validation_seq_fs)
mse_baseline = mean_squared_error(y_validation_seq_fs, y_pred_baseline)

n_repeats = 10
importanza_media = np.zeros(len(predittori))
importanza_std = np.zeros(len(predittori))

for f in range(len(predittori)):
    perdite = []
    for _ in range(n_repeats):
        X_permutato = X_validation_seq_fs.copy()
        perm_idx = np.random.permutation(X_permutato.shape[0])
        X_permutato[:, :, f] = X_permutato[perm_idx][:, :, f]
        y_pred_permutato = lstm_baseline.predict(X_permutato)
        mse_permutato = mean_squared_error(y_validation_seq_fs, y_pred_permutato)
        perdite.append(mse_permutato - mse_baseline)
    importanza_media[f] = np.mean(perdite)
    importanza_std[f] = np.std(perdite)

ranking = pd.DataFrame({
    "Variabile": predittori,
    "Importanza_media": importanza_media,
    "Importanza_std": importanza_std,
})
ranking = ranking.sort_values("Importanza_media", ascending=False)

print("RANKING PERMUTATION IMPORTANCE (LSTM)")
print(ranking)

# Forward feature selection guidata dal ranking (stessa logica del Random Forest)
ranked_features = ranking["Variabile"].tolist()
feature_sizes = list(range(17, 4, -1))

results_features = []

for n in feature_sizes:
    selected_features = ranked_features[:n]
    idx_selected = [predittori.index(f) for f in selected_features]

    lstm_sel = NeuralNetRegressor(
        module=LSTMModel,
        module__input_size=n,
        module__hidden_size=10,
        module__num_layers=2,
        module__dropout=0.3,
        optimizer__lr=0.001,
        batch_size=32,
        max_epochs=50,
        **kwargs_comuni,
    )
    lstm_sel.fit(X_training_seq_fs[:, :, idx_selected], y_training_seq_fs)

    y_pred_sel = lstm_sel.predict(X_validation_seq_fs[:, :, idx_selected])

    rmse = np.sqrt(mean_squared_error(y_validation_seq_fs, y_pred_sel))
    mae = mean_absolute_error(y_validation_seq_fs, y_pred_sel)
    r2 = r2_score(y_validation_seq_fs, y_pred_sel)

    results_features.append({
        "Numero_feature": n,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Features": selected_features,
    })

results_features = pd.DataFrame(results_features)

print("FEATURES SELEZIONATE (LSTM, PERMUTATION IMPORTANCE)")
print(results_features[["Numero_feature", "RMSE", "MAE", "R2"]])

best_features = results_features.loc[results_features["RMSE"].idxmin(), "Features"]
print("Feature selezionate:", best_features)

predittori = list(best_features)  # da qui in avanti si usano solo le feature selezionate


# -----------------------------------------------------
# RI-COSTRUZIONE DEI DATASET SULLE FEATURE SELEZIONATE
# -----------------------------------------------------
X_training = dataset_h.loc[:"2019-12-31", predittori]
X_validation = dataset_h.loc["2020-01-01":"2021-12-31", predittori]
X_test = dataset_h.loc["2022-01-01":, predittori]

X_train_validation = pd.concat([X_training, X_validation])


# -------------------
# SCALING DEFINITIVO
# -------------------
# Fit una sola volta su train + validation, riutilizzato invariato per tutto il backtest (per via dell'elevato costo computazionale di rifarlo ad ogni iterazione del backtest)
scaler_X = RobustScaler()
scaler_y = RobustScaler()
scaler_X.fit(X_train_validation)
scaler_y.fit(y_train_validation.values.reshape(-1, 1))

X_train_validation_scaled = scaler_X.transform(X_train_validation)
y_train_validation_scaled = scaler_y.transform(y_train_validation.values.reshape(-1, 1)).ravel()


# ---------------------------------------------------
# TENSORIZZAZIONE TRAIN + VALIDATION (per il tuning)
# ---------------------------------------------------
X_tv_seq, y_tv_seq = [], []
for t in range(lookback, len(X_train_validation_scaled)):
    X_tv_seq.append(X_train_validation_scaled[t - lookback:t, :])
    y_tv_seq.append(y_train_validation_scaled[t])
X_tv_seq = np.array(X_tv_seq, dtype=np.float32)
y_tv_seq = np.array(y_tv_seq, dtype=np.float32).reshape(-1, 1)

# PredefinedSplit sulle sequenze (le prime "lookback" righe di training si perdono)
test_fold = np.concatenate([
    np.full(len(X_training) - lookback, -1),
    np.full(len(X_validation), 0),
])
ps = PredefinedSplit(test_fold)


# ------------------------------------------
# GRID SEARCH CV OPTIMIZATION IPERPARAMETRI
# ------------------------------------------
param_grid_cross_validation = {
    "module__hidden_size": [8, 16, 32],
    "module__num_layers": [1, 2],
    "module__dropout": [0.2, 0.3],
    "optimizer__lr": [0.001, 0.0001],
    "batch_size": [16, 32],
}

n_iter_grid_search = len(ParameterGrid(param_grid_cross_validation))

lstm_wrapper = NeuralNetRegressor(
    module=LSTMModel,
    module__input_size=len(predittori),
    max_epochs=50,
    **kwargs_comuni,
)

start_cross_validation = time.time()

grid_search = GridSearchCV(
    estimator=lstm_wrapper,
    param_grid=param_grid_cross_validation,
    cv=ps,
    scoring="neg_mean_squared_error",
    n_jobs=1,  # con PyTorch/GPU n_jobs=1 evita conflitti
    verbose=2,
)

grid_search.fit(X_tv_seq, y_tv_seq)
tempo_grid_search = time.time() - start_cross_validation
best_iperparameters_cross_validation = grid_search.best_params_

print("MIGLIORI IPERPARAMETRI GRID-SEARCH CV:", best_iperparameters_cross_validation)
print("TEMPO DI TUNING GRID-SEARCH CV:", tempo_grid_search)


# ------------------------------------
# BAYESIAN OPTIMIZATION IPERPARAMETRI
# ------------------------------------
search_spaces_bayesian_optimization = {
    "module__hidden_size": Categorical([8, 16, 32]),
    "module__num_layers": Categorical([1, 2]),
    "module__dropout": Categorical([0.2, 0.3]),
    "optimizer__lr": Categorical([0.001, 0.0001]),
    "batch_size": Categorical([16, 32]),
}

start_bayesian_optimization = time.time()

bayesian_optimization = BayesSearchCV(
    estimator=lstm_wrapper,
    search_spaces=search_spaces_bayesian_optimization,
    scoring="neg_mean_squared_error",
    cv=ps,
    n_iter=n_iter_grid_search,
    n_jobs=1,
    random_state=42,
    verbose=2,
)

bayesian_optimization.fit(X_tv_seq, y_tv_seq)
tempo_bayesian_optimization = time.time() - start_bayesian_optimization
best_iperparameters_bayesian_optimization = dict(bayesian_optimization.best_params_)

print("MIGLIORI IPERPARAMETRI BAYESIAN OPTIMIZATION:", best_iperparameters_bayesian_optimization)
print("TEMPO DI TUNING BAYESIAN OPTIMIZATION:", tempo_bayesian_optimization)


# ------------------------------------------------------------------
# PREPARAZIONE DATI PER IL BACKTEST (scaler fissato, non rifittato)
# ------------------------------------------------------------------
X_all = pd.concat([X_training, X_validation, X_test])
y_all = pd.concat([y_training, y_validation, y_test])

X_all_scaled = scaler_X.transform(X_all)
y_all_scaled = scaler_y.transform(y_all.values.reshape(-1, 1)).ravel()

n_train = len(X_training)
n_validation = len(X_validation)
n_test = len(X_test)


# ---------------------------------------------------------------
# SLIDING WINDOW BACKTEST (WALK-FORWARD FORECAST) GRID-SEARCH CV
# ---------------------------------------------------------------
y_predicted_list_cv = []
dates_predicted_cv = []

start = time.time()

for i in range(n_test):
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"Iterazione {i}/{n_test} - {elapsed:.1f} s")

    train_window_X_scaled = X_all_scaled[i: i + n_train + n_validation]
    train_window_y_scaled = y_all_scaled[i: i + n_train + n_validation]

    # Tensorizzazione della finestra di training corrente
    X_window_seq, y_window_seq = [], []
    for t in range(lookback, len(train_window_X_scaled)):
        X_window_seq.append(train_window_X_scaled[t - lookback:t, :])
        y_window_seq.append(train_window_y_scaled[t])
    X_window_seq = np.array(X_window_seq, dtype=np.float32)
    y_window_seq = np.array(y_window_seq, dtype=np.float32).reshape(-1, 1)

    target_idx = i + n_train + n_validation
    X_new_seq = X_all_scaled[target_idx - lookback: target_idx, :].reshape(1, lookback, len(predittori)).astype(np.float32)

    lstm_cv_backtest = NeuralNetRegressor(
        module=LSTMModel,
        module__input_size=len(predittori),
        max_epochs=50,
        **kwargs_comuni,
        **best_iperparameters_cross_validation,
    )
    lstm_cv_backtest.fit(X_window_seq, y_window_seq)

    pred_scaled_cv = lstm_cv_backtest.predict(X_new_seq)[0][0]
    pred_cv = scaler_y.inverse_transform([[pred_scaled_cv]])[0][0]

    y_predicted_list_cv.append(pred_cv)
    dates_predicted_cv.append(X_all.index[target_idx])

# Riallineamento valori previsti y con date
y_predicted_backtest_cv = pd.Series(y_predicted_list_cv, index=dates_predicted_cv, name="VIX_Forecasted")
y_true_backtest_cv = y_all.loc[y_predicted_backtest_cv.index].rename("VIX_Reale")

# Cartella comune dove vengono salvati i risultati
output_dir = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\3. Deep Learning\1. LSTM\Results\0_Forecast"
os.makedirs(output_dir, exist_ok=True)

df_out_cv = pd.DataFrame({
    "y_true": y_true_backtest_cv,
    "y_pred": y_predicted_backtest_cv
})
df_out_cv.to_csv(os.path.join(output_dir, f"lstm_gridsearch_h{orizzonte}.csv"))


# ------------------------
# CALCOLO METRICHE ERRORE
# ------------------------
mse_cv = mean_squared_error(y_true_backtest_cv, y_predicted_backtest_cv)
mae_cv = mean_absolute_error(y_true_backtest_cv, y_predicted_backtest_cv)
mape_cv = mean_absolute_percentage_error(y_true_backtest_cv, y_predicted_backtest_cv)
r2_cv = r2_score(y_true_backtest_cv, y_predicted_backtest_cv)
qlike_cv = np.mean((y_true_backtest_cv / y_predicted_backtest_cv) - np.log(y_true_backtest_cv / y_predicted_backtest_cv) - 1)

actual_direction_cv = np.sign(y_true_backtest_cv.diff())
predicted_direction_cv = np.sign(y_predicted_backtest_cv - y_true_backtest_cv.shift(1))
directional_accuracy_cv = (actual_direction_cv == predicted_direction_cv).iloc[1:].mean() * 100

print("\n--- METRICHE BACKTEST SLIDING WINDOW LSTM GRID-SEARCH CV ---")
print(f"MSE {orizzonte}:  {mse_cv:.4f}")
print(f"MAE:  {mae_cv:.4f}")
print(f"MAPE: {mape_cv:.4f}")
print(f"R^2:  {r2_cv:.4f}")
print(f"QLIKE: {qlike_cv:.4f}")
print(f"Directional Accuracy: {directional_accuracy_cv:.2f}%")


# -----------------------------------------------------------------------
# SLIDING WINDOW BACKTEST (WALK-FORKWARD FORECAST) BAYESIAN OPTIMIZATION
# -----------------------------------------------------------------------
y_predicted_list_bo = []
dates_predicted_bo = []

start = time.time()

for i in range(n_test):
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"Iterazione {i}/{n_test} - {elapsed:.1f} s")

    train_window_X_scaled = X_all_scaled[i: i + n_train + n_validation]
    train_window_y_scaled = y_all_scaled[i: i + n_train + n_validation]

    X_window_seq, y_window_seq = [], []
    for t in range(lookback, len(train_window_X_scaled)):
        X_window_seq.append(train_window_X_scaled[t - lookback:t, :])
        y_window_seq.append(train_window_y_scaled[t])
    X_window_seq = np.array(X_window_seq, dtype=np.float32)
    y_window_seq = np.array(y_window_seq, dtype=np.float32).reshape(-1, 1)

    target_idx = i + n_train + n_validation
    X_new_seq = X_all_scaled[target_idx - lookback: target_idx, :].reshape(1, lookback, len(predittori)).astype(np.float32)

    lstm_bo_backtest = NeuralNetRegressor(
        module=LSTMModel,
        module__input_size=len(predittori),
        max_epochs=50,
        **kwargs_comuni,
        **best_iperparameters_bayesian_optimization,
    )
    lstm_bo_backtest.fit(X_window_seq, y_window_seq)

    pred_scaled_bo = lstm_bo_backtest.predict(X_new_seq)[0][0]
    pred_bo = scaler_y.inverse_transform([[pred_scaled_bo]])[0][0]

    y_predicted_list_bo.append(pred_bo)
    dates_predicted_bo.append(X_all.index[target_idx])

y_predicted_backtest_bo = pd.Series(y_predicted_list_bo, index=dates_predicted_bo, name="VIX_Forecasted")
y_true_backtest_bo = y_all.loc[y_predicted_backtest_bo.index].rename("VIX_Reale")

df_out_bo = pd.DataFrame({
    "y_true": y_true_backtest_bo,
    "y_pred": y_predicted_backtest_bo
})
df_out_bo.to_csv(os.path.join(output_dir, f"lstm_bayesoptimization_h{orizzonte}.csv"))


# ------------------------
# CALCOLO METRICHE ERRORE
# ------------------------
mse_bo = mean_squared_error(y_true_backtest_bo, y_predicted_backtest_bo)
mae_bo = mean_absolute_error(y_true_backtest_bo, y_predicted_backtest_bo)
mape_bo = mean_absolute_percentage_error(y_true_backtest_bo, y_predicted_backtest_bo)
r2_bo = r2_score(y_true_backtest_bo, y_predicted_backtest_bo)
qlike_bo = np.mean((y_true_backtest_bo / y_predicted_backtest_bo) - np.log(y_true_backtest_bo / y_predicted_backtest_bo) - 1)

actual_direction_bo = np.sign(y_true_backtest_bo.diff())
predicted_direction_bo = np.sign(y_predicted_backtest_bo - y_true_backtest_bo.shift(1))
directional_accuracy_bo = (actual_direction_bo == predicted_direction_bo).iloc[1:].mean() * 100

print("\n--- METRICHE BACKTEST SLIDING WINDOW LSTM BAYESIAN OPTIMIZATION ---")
print(f"MSE {orizzonte}: {mse_bo:.4f}")
print(f"MAE: {mae_bo:.4f}")
print(f"MAPE: {mape_bo:.4f}")
print(f"R^2: {r2_bo:.4f}")
print(f"QLIKE: {qlike_bo:.4f}")
print(f"Directional Accuracy: {directional_accuracy_bo:.2f}%")


# -------------------------------------
# GRAFICI: VIX REALE vs VIX FORECASTED
# -------------------------------------
y_true_backtest_cv.index = pd.to_datetime(y_true_backtest_cv.index)
y_predicted_backtest_cv.index = pd.to_datetime(y_predicted_backtest_cv.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true_backtest_cv.index, y_true_backtest_cv, label="VIX Reale", color="black", linewidth=1.2)
ax.plot(y_predicted_backtest_cv.index, y_predicted_backtest_cv, label="VIX Previsto (LSTM)", color="red", linewidth=1.2, alpha=0.8)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=45)

output_dir_grafici = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\3. Deep Learning\1. LSTM\Results\1_Grafici_backtest"
os.makedirs(output_dir_grafici, exist_ok=True)

ax.set_title(f"VIX Reale vs VIX Previsto (Test Set) con ottimizzazione iperparametri tramite Grid Search CV h = {orizzonte}")
ax.set_xlabel("Data")
ax.set_ylabel("VIX")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir_grafici, f"lstm_CV_vix_reale_vs_previsto_h{orizzonte}.png"), dpi=300, bbox_inches="tight")
plt.show()

y_true_backtest_bo.index = pd.to_datetime(y_true_backtest_bo.index)
y_predicted_backtest_bo.index = pd.to_datetime(y_predicted_backtest_bo.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true_backtest_bo.index, y_true_backtest_bo, label="VIX Reale", color="black", linewidth=1.2)
ax.plot(y_predicted_backtest_bo.index, y_predicted_backtest_bo, label="VIX Previsto (LSTM)", color="red", linewidth=1.2, alpha=0.8)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=45)

ax.set_title(f"VIX Reale vs VIX Previsto (Test Set) con Bayesian Optimization h = {orizzonte}")
ax.set_xlabel("Data")
ax.set_ylabel("VIX")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir_grafici, f"lstm_BO_vix_reale_vs_previsto_h{orizzonte}.png"), dpi=300, bbox_inches="tight")
plt.show()