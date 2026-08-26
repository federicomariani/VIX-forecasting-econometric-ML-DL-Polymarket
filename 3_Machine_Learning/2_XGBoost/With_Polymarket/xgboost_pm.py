from xgboost import XGBRegressor
import pandas as pd
import numpy as np
np.int = int # serve per permettere a skop di funzionare

import shap
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
from skopt.space import Categorical # per Bayesian Optimization
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid 



# ---------------
# IMPORT DATASET
# ---------------
dataset = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\df_finale.csv", index_col=["Date"])


# -----------------
# DATA PREPARATION
# -----------------
# Conversione Date -> DatetimeIndex
dataset.index = pd.to_datetime(
    dataset.index,
    errors="coerce"
)

# Orizzonte di previsione 1, 5 e 22 giorni
orizzonte = 1

# Con o senza PM
evento = "senza PM" # Da cambiare quando si utilizzano le features di Polymarket

# Creazione target shiftato in avanti di h giorni (previsione diretta multi-step)
target_col = f"Close_VIX_h{orizzonte}"
dataset[target_col] = dataset["Close_VIX"].shift(-orizzonte)

# Rinomino colonne con segni che non vengono accettati dalla funzione
dataset = dataset.rename(columns={"cpi_<=2.7": "cpi_max_2_7",
                                  "cpi_2.8": "cpi_2_8",
                                  "cpi_2.9": "cpi_2_9",
                                  "cpi_>=3.0": "cpi_min_3_0"})

# Rimuovo le ultime righe che non hanno un target valido (fine serie)
dataset_h = dataset.dropna(subset=[target_col])

# Creazione matrice variabili indipendenti e vettore variabile dipendente 
predittori = ["Log_Return_S&P500", "Log_Return_WTI", "Close_OVX", "OVX_VIX_spread",
              "VIX_lag1", "VIX_lag5", "VIX_lag22", "RSI_14d", "Log_Return_DXY",
              "Yield_curve", "Log_difference_initial_claims", "VIX_MA5", "VIX_MA10", "VIX_MA20",
              "Credit_Spread", "Log_Return_Gold", "Weekday"] 

predittori_w_pm = ["Log_Return_S&P500", "Log_Return_WTI", "Close_OVX", "OVX_VIX_spread",
              "VIX_lag1", "VIX_lag5", "VIX_lag22", "RSI_14d", "Log_Return_DXY",
              "Yield_curve", "Log_difference_initial_claims", "VIX_MA5", "VIX_MA10", "VIX_MA20",
              "Credit_Spread", "Log_Return_Gold", "Weekday", "cut_50", "cut_25", "hold", "hike_25",
              "expected_rate_change_fomc", "rate_uncertainty_fomc", "probability_cut_fomc", 
              "probability_hike_fomc", "entropy_fomc", "us_recession", "entropy_us_recession",
              "change_1d_us_recession", "change_5d_us_recession", "cpi_max_2_7", "cpi_2_8",
              "cpi_2_9", "cpi_min_3_0", "expected_cpi", "uncertainty_cpi", "entropy_cpi"] # predittori with polymarket features

target = [target_col] # Variabile dipendente


# ------------------
# DATA MANIPULATION
# ------------------
# Suddivisione matrici in training, validation e test set (prima faccio senza Polymarket features, poi sostituisco e ci metto predittori_w_pm)
X_training = dataset_h.loc[:"2019-12-31", predittori]
y_training = dataset_h.loc[:"2019-12-31", target].squeeze() # uso squeeze() perché y deve essere una lista affinché possa essere usata nella funzione XGBoost

X_validation = dataset_h.loc["2020-01-01":"2021-12-31", predittori]
y_validation = dataset_h.loc["2020-01-01":"2021-12-31", target].squeeze()

X_test = dataset_h.loc["2022-01-01":, predittori]
y_test = dataset_h.loc["2022-01-01":, target].squeeze()

# Unione del train e del validation set in quanto richiesto da PredefinedSplit 
X_train_validation = pd.concat([X_training, X_validation])
y_train_validation = pd.concat([y_training, y_validation])

# Si dice a PredefinedSplit quali righe usare per fare training del modello (-1) e quali per validation (0)
test_fold = np.concatenate([
    np.full(len(X_training), -1),      # righe di training --> escluse dalla validazione
    np.full(len(X_validation), 0)   # righe di validation --> usate per validare
])

ps = PredefinedSplit(test_fold)


# ------------------------------------------------------------------------
# SELEZIONE FEATURES TRAMITE BACKWARD FEATURE SELECTION (GUIDATA DA SHAP)
# ------------------------------------------------------------------------
# Features rankate da SHAP 
model = XGBRegressor(
    random_state = 42,
    n_estimators = 200,
    max_depth = 3,
    learning_rate = 0.1,
    objective="reg:pseudohubererror"
)

model.fit(X_training, y_training)

explainer = shap.Explainer(model)
shap_values = explainer(X_training)

importance = pd.DataFrame({
    "Feature": X_training.columns,
    "Importance": np.abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("FEATURES RANKATE DA SHAP")
print(importance)

# Features scelte tramite metodo Backward Feature Selection guidata da shap
ranked_features = importance["Feature"].tolist()
print(ranked_features)


# ------------------------------------------
# GRID SEARCH CV OPTIMIZATION IPERPARAMETRI
# ------------------------------------------
param_grid_cross_validation = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "reg_alpha": [0, 0.1, 1], # alpha
    "reg_lambda": [1, 5, 10] # lambda
}

n_iter_grid_search = len(ParameterGrid(param_grid_cross_validation))

start_cross_validation = time.time()

grid_search = GridSearchCV(
    estimator=XGBRegressor(
        objective="reg:pseudohubererror",
        random_state=42
    ),
    param_grid=param_grid_cross_validation,
    scoring="neg_root_mean_squared_error",
    n_jobs= -1,
    cv=ps
)

grid_search.fit(X_train_validation, y_train_validation)
tempo_grid_search = time.time() - start_cross_validation
best_iperparameters_cross_validation = grid_search.best_params_

print("Migliori iperparametri Grid-Search CV:", best_iperparameters_cross_validation)
print("Tempo di tuning di Grid-Search CV:", tempo_grid_search)


# ------------------------------------
# BAYESIAN OPTIMIZATION IPERPARAMETRI
# ------------------------------------
search_spaces_bayesian_optimization = {
    "n_estimators": Categorical([100, 200, 300, 500]),
    "max_depth": Categorical([3, 5, 7]),
    "learning_rate": Categorical([0.01, 0.05, 0.1]),
    "reg_alpha": Categorical([0, 0.1, 1]),
    "reg_lambda": Categorical([1, 5, 10]),
}

start_bayesian_optimization = time.time()

bayesian_optimization = BayesSearchCV(
    estimator = XGBRegressor(
        objective = "reg:pseudohubererror",
        random_state = 42
    ),
    search_spaces = search_spaces_bayesian_optimization,
    scoring = "neg_root_mean_squared_error",
    n_jobs = -1,
    cv=ps,
    n_iter=n_iter_grid_search,
    random_state = 42
)

bayesian_optimization.fit(X_train_validation, y_train_validation)
tempo_bayesian_optimization = time.time() - start_bayesian_optimization
best_iperparameters_bayesian_optimization = dict(bayesian_optimization.best_params_)

print("Migliori iperparametri Bayesian Optimization:", best_iperparameters_bayesian_optimization) 
print("Tempo di tuning di Bayesian Optimization:", tempo_bayesian_optimization)

# ------------
# SHAP VALUES 
# ------------
# Features rankate da shap dopo ottimizzazione Grid Search CV
model_cv = XGBRegressor(
    **grid_search.best_params_,
    objective="reg:pseudohubererror",
    random_state = 42
)

model_cv.fit(X_training, y_training)

explainer_cv = shap.Explainer(model_cv)
shap_values_cv = explainer_cv(X_validation)

importance_cv = pd.DataFrame({
    "Feature": X_validation.columns,
    "Importance": np.abs(shap_values_cv.values).mean(axis=0)
})

importance_cv = importance_cv.sort_values(
    "Importance",
    ascending=False
)

print("FEATURES RANKATE DA SHAP DOPO OTTIMIZZAZIONE GRID SEARCH CV")
print(importance_cv)

# Features rankate da shap dopo Bayesian Optimization 
model_bo = XGBRegressor(
    **best_iperparameters_bayesian_optimization,
    objective="reg:pseudohubererror",
    random_state=42
)

model_bo.fit(X_training, y_training)

explainer_bo = shap.Explainer(model_bo)
shap_values_bo = explainer_bo(X_validation)

importance_bo = pd.DataFrame({
    "Feature": X_validation.columns,
    "Importance": np.abs(shap_values_bo.values).mean(axis=0)
})

importance_bo = importance_bo.sort_values(
    "Importance",
    ascending=False
)

print("FEATURES RANKATE DA SHAP DOPO BAYESIAN OPTIMIZATION")
print(importance_bo)


# ------------------------------------------------------------------------------
# SELEZIONE FEATURES TRAMITE METODO BACKWARD FEATURE SELECTION (GUIDATA DA SHAP)
# ------------------------------------------------------------------------------
# Selezione features da shap dopo Grid Search CV
ranked_features_cv = importance_cv["Feature"].tolist()
feature_sizes_cv = list(range(17, 4, -1)) # 17 = feature meno importante

results_features_cv = []

for n in feature_sizes_cv:
    # prendo le n migliori feature secondo SHAP
    selected_features_cv = ranked_features_cv[:n]

    X_training_selected = X_training[selected_features_cv]
    X_validation_selected = X_validation[selected_features_cv]

    model_sel_cv = XGBRegressor(
        **grid_search.best_params_,
        objective="reg:pseudohubererror",
        random_state=42
    )

    model_sel_cv.fit(X_training_selected, y_training)

    y_pred_sel = model_sel_cv.predict(X_validation_selected)

    rmse = np.sqrt(mean_squared_error(y_validation, y_pred_sel))

    mae = mean_absolute_error(y_validation, y_pred_sel)

    r2 = r2_score(y_validation, y_pred_sel)

    results_features_cv.append({
        "Numero_feature": n,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Features": selected_features_cv
    })

results_features_cv = pd.DataFrame(results_features_cv)

print("FEATURES SELEZIONATE DA SHAP DOPO OTTIMIZZAZIONE GRID SEARCH CV")
print(results_features_cv[[
    "Numero_feature",
    "RMSE",
    "MAE",
    "R2"
]])

best_features_cv = results_features_cv.loc[
    results_features_cv["RMSE"].idxmin(),
    "Features"
]

print(best_features_cv)

# Selezione features da SHAP dopo Bayesian Optimization
ranked_features_bo = importance_bo["Feature"].tolist()
feature_sizes_bo = list(range(17, 4, -1))

results_features_bo = []

for n in feature_sizes_bo:
    # prendo le n migliori feature secondo SHAP
    selected_features_bo = ranked_features_bo[:n]

    X_training_selected = X_training[selected_features_bo]
    X_validation_selected = X_validation[selected_features_bo]

    model_sel_bo = XGBRegressor(
        **best_iperparameters_bayesian_optimization,
        objective="reg:pseudohubererror",
        random_state=42
    )

    model_sel_bo.fit(X_training_selected, y_training)

    y_pred_sel = model_sel_bo.predict(X_validation_selected)

    rmse = np.sqrt(mean_squared_error(y_validation, y_pred_sel))

    mae = mean_absolute_error(y_validation, y_pred_sel)

    r2 = r2_score(y_validation, y_pred_sel)

    results_features_bo.append({
        "Numero_feature": n,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Features": selected_features_bo
    })

results_features_bo = pd.DataFrame(results_features_bo)

print("FEATURES SELEZIONATE DA SHAP DOPO BAYESIAN OPTIMIZATION")
print(results_features_bo[[
    "Numero_feature",
    "RMSE",
    "MAE",
    "R2"
]])

best_features_bo = results_features_bo.loc[
    results_features_bo["RMSE"].idxmin(),
    "Features"
]

print(best_features_bo)


# ----------------------------------------------------------------
# SLIDING WINDOW BACKTEST (WALK-FORKWARD FORECAST) GRID-SEARCH CV
# ----------------------------------------------------------------
# Si riuniscono i tre dataset in maniera cronologica per poter effettuare quesa tipologia di backtest
X_all = pd.concat([X_training, X_validation, X_test])
y_all = pd.concat([y_training, y_validation, y_test])

n_train = len(X_training)
n_validation = len(X_validation)
n_test = len(X_test)


# Inizio ciclo for del backtest. cv = Cross Validation
y_predicted_list_cv = []
dates_predicted_cv = []

start = time.time()

for i in range(n_test):
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"Iterazione {i}/{n_test} - {elapsed:.1f} s")
    
    train_window_X = X_all.iloc[i : i + n_train + n_validation][best_features_cv]
    train_window_y = y_all.iloc[i : i + n_train + n_validation]
    
    target_idx = i + n_train + n_validation
    X_new = X_all.iloc[[target_idx]][best_features_cv]
    
    xgboost_cv_backtest = XGBRegressor(
        **best_iperparameters_cross_validation,
         n_jobs=-1,
        random_state=42,
    )
    xgboost_cv_backtest.fit(train_window_X, train_window_y)

    pred_cv = xgboost_cv_backtest.predict(X_new)[0]
    y_predicted_list_cv.append(pred_cv)
    dates_predicted_cv.append(X_all.index[target_idx])

# Riallineamento valori previsti y con date
y_predicted_backtest_cv = pd.Series(y_predicted_list_cv, index=dates_predicted_cv, name="VIX_Forecasted")
y_true_backtest_cv = y_all.loc[y_predicted_backtest_cv.index].rename("VIX_Reale")

# Cartella comune dove tutti i file-modello salvano i risultati
output_dir = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\2. XGBoost\Results\0_Forecast"
os.makedirs(output_dir, exist_ok=True)

# Salvo previsioni + valori reali, per la variante Grid-Search per fare poi Model Confidence Set e Diebold-Mariano test
df_out_cv = pd.DataFrame({
    "y_true": y_true_backtest_cv,
    "y_pred": y_predicted_backtest_cv
})
df_out_cv.to_csv(os.path.join(output_dir, f"xgboost_gridsearch_h{orizzonte}.csv"))


# ------------------------
# CALCOLO METRICHE ERRORE
# ------------------------
mse_cv = mean_squared_error(y_true_backtest_cv, y_predicted_backtest_cv)
mae_cv = mean_absolute_error(y_true_backtest_cv, y_predicted_backtest_cv)
mape_cv = mean_absolute_percentage_error(y_true_backtest_cv, y_predicted_backtest_cv)
r2_cv = r2_score(y_true_backtest_cv, y_predicted_backtest_cv)
qlike_cv = np.mean((y_true_backtest_cv / y_predicted_backtest_cv) - np.log(y_true_backtest_cv / y_predicted_backtest_cv) -1)

actual_direction_cv = np.sign(y_true_backtest_cv.diff())
predicted_direction_cv = np.sign(y_predicted_backtest_cv - y_true_backtest_cv.shift(1))
directional_accuracy_cv = (actual_direction_cv == predicted_direction_cv).iloc[1:].mean() * 100

print("\n--- METRICHE BACKTEST SLIDING WINDOW XGBOOST GRID-SEARCH CV ---")
print(f"MSE {orizzonte}:  {mse_cv:.4f}")
print(f"MAE:  {mae_cv:.4f}")
print(f"MAPE: {mape_cv:.4f}")
print(f"R^2:  {r2_cv:.4f}")
print(f"QLIKE: {qlike_cv:.4f}")
print(f"Directional Accuracy: {directional_accuracy_cv:.2f}%")

# -----------------------------------------------------------------------
# SLIDING WINDOW BACKTEST (WALK-FORKWARD FORECAST) BAYESIAN OPTIMIZATION
# -----------------------------------------------------------------------
# Inizio ciclo for del backtest. bo = Bayesian Optimization
y_predicted_list_bo = []
dates_predicted_bo = []

start = time.time()

for i in range(n_test):
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"Iterazione {i}/{n_test} - {elapsed:.1f} s")
    
    train_window_X = X_all.iloc[i : i + n_train + n_validation][best_features_bo]
    train_window_y = y_all.iloc[i : i + n_train + n_validation]
    
    target_idx = i + n_train + n_validation
    X_new = X_all.iloc[[target_idx]][best_features_bo]
    
    xgboost_bo_backtest = XGBRegressor(
        **bayesian_optimization.best_params_,
         n_jobs=-1,
        random_state=42,
    )
    xgboost_bo_backtest.fit(train_window_X, train_window_y)

    pred_bo = xgboost_bo_backtest.predict(X_new)[0]
    y_predicted_list_bo.append(pred_bo)
    dates_predicted_bo.append(X_all.index[target_idx])

# Riallineamento valori previsti y con date
y_predicted_backtest_bo = pd.Series(y_predicted_list_bo, index=dates_predicted_bo, name="VIX_Forecasted")
y_true_backtest_bo = y_all.loc[y_predicted_backtest_bo.index].rename("VIX_Reale")

# Salvo previsioni + valori reali, per la variante Bayesian Optimization
df_out_bo = pd.DataFrame({
    "y_true": y_true_backtest_bo,
    "y_pred": y_predicted_backtest_bo
})
df_out_bo.to_csv(os.path.join(output_dir, f"xgboost_bayesoptimization_h{orizzonte}.csv"))


# ------------------------
# CALCOLO METRICHE ERRORE
# ------------------------
mse_bo = mean_squared_error(y_true_backtest_bo, y_predicted_backtest_bo)
mae_bo = mean_absolute_error(y_true_backtest_bo, y_predicted_backtest_bo)
mape_bo = mean_absolute_percentage_error(y_true_backtest_bo, y_predicted_backtest_bo)
r2_bo = r2_score(y_true_backtest_bo, y_predicted_backtest_bo)
qlike_bo = np.mean((y_true_backtest_bo / y_predicted_backtest_bo) - np.log(y_true_backtest_bo / y_predicted_backtest_bo) -1)

actual_direction_bo = np.sign(y_true_backtest_bo.diff())
predicted_direction_bo = np.sign(y_predicted_backtest_bo - y_true_backtest_bo.shift(1))
directional_accuracy_bo = (actual_direction_bo == predicted_direction_bo).iloc[1:].mean() * 100

print("\n--- METRICHE BACKTEST SLIDING WINDOW XGBOOST BAYESIAN OPTIMIZATION ---")
print(f"MSE {orizzonte}: {mse_bo:.4f}")
print(f"MAE: {mae_bo:.4f}")
print(f"MAPE: {mape_bo:.4f}")
print(f"R^2: {r2_bo:.4f}")
print(f"QLIKE: {qlike_bo:.4f}")
print(f"Directional Accuracy: {directional_accuracy_bo:.2f}%")


# -------------------------------------
# GRAFICI: VIX REALE vs VIX FORECASTED  
# -------------------------------------
# Grafico con ottimizzazione Grid-Search CV
# Riportare l'indice in formato datetime (per asse delle x leggibile) 
y_true_backtest_cv.index = pd.to_datetime(y_true_backtest_cv.index)
y_predicted_backtest_cv.index = pd.to_datetime(y_predicted_backtest_cv.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true_backtest_cv.index, y_true_backtest_cv, label="VIX Reale", color="black", linewidth=1.2)
ax.plot(y_predicted_backtest_cv.index, y_predicted_backtest_cv, label="VIX Previsto (XGBoost)", color="red", linewidth=1.2, alpha=0.8)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))   # un tick ogni mese
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))   # formato tipo "Gen 2022"
fig.autofmt_xdate(rotation=45)                                # ruota le etichette per non sovrapporle

ax.set_title(f"VIX Reale vs VIX Previsto (Test Set) con ottimizzazione iperparametri tramite Grid Search CV h = {orizzonte}")
ax.set_xlabel("Data")
ax.set_ylabel("VIX")
ax.legend()
plt.tight_layout()
plt.savefig(rf"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\2. XGBoost\Results\xgboost_CV_vix_reale_vs_previsto_h{orizzonte}.png", dpi=300, bbox_inches="tight")
plt.show()


# Grafico con Bayesian Optimization
# Riportare l'indice in formato datetime (per asse delle x leggibile) 
y_true_backtest_bo.index = pd.to_datetime(y_true_backtest_bo.index)
y_predicted_backtest_bo.index = pd.to_datetime(y_predicted_backtest_bo.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true_backtest_bo.index, y_true_backtest_bo, label="VIX Reale", color="black", linewidth=1.2)
ax.plot(y_predicted_backtest_bo.index, y_predicted_backtest_bo, label="VIX Previsto (XGBoost)", color="red", linewidth=1.2, alpha=0.8)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))   
fig.autofmt_xdate(rotation=45)                                

ax.set_title(f"VIX Reale vs VIX Previsto (Test Set) con Bayesian Optimization h = {orizzonte}")
ax.set_xlabel("Data")
ax.set_ylabel("VIX")
ax.legend()
plt.tight_layout()
plt.savefig(rf"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\2. XGBoost\Results\xgboost_BO_vix_reale_vs_previsto_h{orizzonte}.png", dpi=300, bbox_inches="tight")
plt.show()
