# This model is identical to the model in the file "random_forest",
# with the only difference that here all VIX lags up to 22 days are considered,
# whereas in the other model only the VIX lags at 1, 5 and 22 business days are used



import os
import pandas as pd
import numpy as np
np.int = int # needed to allow skopt to work

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid # import this library to preserve
                                                                                 # the chronological order of the time series,
                                                                                 # which would otherwise be lost with a standard CV,
                                                                                 # since it would mix dates and violate time ordering

# ---------------
# IMPORT DATASET
# ---------------
dataset = pd.read_csv(r"C:\Users\fede1\Desktop\Repo\0_Dataset\Data\Clean\dataset.csv", index_col=["Date"])


# -----------------
# DATA PREPARATION
# -----------------
# Add the lags from 1 to 22 business days
vix_lags = [f"VIX_lag{lag}" for lag in range(1, 23)]
for lag in range(1, 23):
    dataset[f"VIX_lag{lag}"] = dataset["Close_VIX"].shift(lag)

# Forecast horizon 1, 5 and 22 days
orizzonte = 22 # Change manually for each run

# Create target shifted forward by h days (direct multi-step forecasting)
target_col = f"Close_VIX_h{orizzonte}"
dataset[target_col] = dataset["Close_VIX"].shift(-orizzonte)

# The row preserves the feature date; the h-step target instead belongs to the observation made h rows later. We keep it to label forecasts and actual values correctly in backtest results
target_date_col = f"Target_Date_h{orizzonte}"
dataset[target_date_col] = dataset.index.to_series().shift(-orizzonte)

# Create the matrix of independent variables and the dependent-variable vector
predittori = ["Log_Return_S&P500", "Log_Return_WTI", "Close_OVX", "OVX_VIX_spread",
              *vix_lags, "RSI_14d", "Log_Return_DXY", "Yield_curve", 
              "Log_difference_initial_claims", "VIX_MA5", "VIX_MA10", "VIX_MA22",
              "Credit_Spread", "Log_Return_Gold", "Weekday"] 
target = [target_col] # Dependent variable


# ------------------
# DATA MANIPULATION
# ------------------
# Remove the last rows that do not have a valid target (end of the series)
dataset_h = dataset.dropna(subset=predittori + [target_col])

# Split matrices into training, validation and test sets
X = dataset_h[predittori]
y = dataset_h[target_col]

n_observations = len(X)

train_end = int(n_observations * 0.60)
validation_end = int(n_observations * 0.70)

X_training = X.iloc[:train_end]
X_validation = X.iloc[train_end:validation_end]
X_test = X.iloc[validation_end:]

y_training = y.iloc[:train_end].squeeze()
y_validation = y.iloc[train_end:validation_end].squeeze()
y_test = y.iloc[validation_end:].squeeze()

print(f"Training set: {len(X_training)} observations ({len(X_training) / n_observations:.2%})")
print(f"Validation set: {len(X_validation)} observations ({len(X_validation) / n_observations:.2%})")
print(f"Test set: {len(X_test)} observations ({len(X_test) / n_observations:.2%})")

for nome, dataset_split in [
    ("Training", X_training),
    ("Validation", X_validation),
    ("Test", X_test)
]:
    print(f"\n{nome} set:")
    print(f"First element: {dataset_split.index[0]}")
    print(f"Last element: {dataset_split.index[-1]}")

# Merge the training and validation sets as required by PredefinedSplit
X_train_validation = pd.concat([X_training, X_validation])
y_train_validation = pd.concat([y_training, y_validation])

# Tell PredefinedSplit which rows to use for model training (-1) and which for validation (0)
test_fold = np.concatenate([
    np.full(len(X_training), -1),      # training rows --> excluded from validation
    np.full(len(X_validation), 0)   # validation rows --> used for validation
])

ps = PredefinedSplit(test_fold)

# ----------------------------------------------
# BUILDING MODEL + VARIABLE IMPORTANCE + BORUTA
# ----------------------------------------------
# Build the first Random Forest Regressor model, which will then be optimized through Boruta and CV
random_forest_1 = RandomForestRegressor(
    n_estimators = 200, # number of trees
    max_features = "sqrt", # square root of all features, as suggested by the literature
    random_state = 42,
)

random_forest_1.fit(X_training, y_training) # train the model

# Variable importance according to MDI (Mean Decrease Impurity)
importance = pd.DataFrame({
    "Variable": X_training.columns,
    "Importance": random_forest_1.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("FEATURE IMPORTANCE BEFORE HYPERPARAMETER OPTIMIZATION")
print(f"Variable importance:{importance}")

# Build the Boruta algorithm to rank variable importance
rf_for_boruta = clone(random_forest_1)

boruta = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators="auto",
    random_state=42,
    verbose=2,
)

boruta.fit(X_training.values, y_training.values) # values because Boruta works with arrays

ranking = pd.DataFrame({
    "Variable": X_training.columns,
    "Rank": boruta.ranking_,
    "Selected": boruta.support_,
    "Tentative": boruta.support_weak_
})

print("BORUTA FEATURE RANKING BEFORE HYPERPARAMETER OPTIMIZATION")
print(f"Variable ranking:{ranking.sort_values('Rank')}")


# -------------------------------------------
# GRID SEARCH CV HYPERPARAMETER OPTIMIZATION
# -------------------------------------------
param_grid_cross_validation = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_features": ["sqrt", "log2"], 
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}

n_iter_grid_search = len(ParameterGrid(param_grid_cross_validation))

start_cross_validation = time.time()

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid_cross_validation,
    cv=ps,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    refit=True,
    return_train_score=True,
    verbose=2,
)

grid_search.fit(X_train_validation, y_train_validation)
tempo_grid_search = time.time() - start_cross_validation
best_iperparameters_cross_validation = grid_search.best_params_

print("BEST GRID-SEARCH CV HYPERPARAMETERS:", best_iperparameters_cross_validation)
print("GRID-SEARCH CV TUNING TIME:", tempo_grid_search)


# --------------------------------------
# BAYESIAN OPTIMIZATION HYPERPARAMETERS
# --------------------------------------
search_spaces_bayesian_optimization = {
    "n_estimators": Integer(100, 500),
    "max_features": Categorical(["sqrt", "log2"]),
    "max_depth": Integer(10, 50),
    "min_samples_leaf": Integer(1, 5),
}

n_iter_bayesian = 100

start_bayesian_optimization = time.time()

bayesian_optimization = BayesSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    search_spaces=search_spaces_bayesian_optimization,
    scoring="neg_mean_squared_error",
    cv=ps,
    n_iter=n_iter_bayesian,
    n_jobs=-1,
    random_state=42,
    refit=True,
    return_train_score=True,
    verbose=2,
)

bayesian_optimization.fit(X_train_validation, y_train_validation)
tempo_bayesian_optimization = time.time() - start_bayesian_optimization
best_iperparameters_bayesian_optimization = dict(bayesian_optimization.best_params_)

print("BEST BAYESIAN OPTIMIZATION HYPERPARAMETERS:", best_iperparameters_bayesian_optimization) 
print("BAYESIAN OPTIMIZATION TUNING TIME:", tempo_bayesian_optimization)


# -------------------------------------------------------------------------------------
# FEATURE IMPORTANCE (MDI & BORUTA) AFTER HYPERPARAMETER OPTIMIZATION - GRID SEARCH CV
# -------------------------------------------------------------------------------------
# Feature importance after hyperparameter optimization via Grid Search CV
model_cv = RandomForestRegressor(
    **grid_search.best_params_,
    random_state = 42
)

model_cv.fit(X_training, y_training)

# Variable importance according to MDI (Mean Decrease Impurity)
importance_mdi_cv = pd.DataFrame({
    "Variable": X_training.columns,
    "Importance": model_cv.feature_importances_
})

importance_mdi_cv = importance_mdi_cv.sort_values(
    by="Importance",
    ascending=False
)

print("FEATURE IMPORTANCE AFTER HYPERPARAMETER OPTIMIZATION - GRID SEARCH CV:")
print(f"Feature importance after grid search CV optimization: {importance_mdi_cv}")

# Build the Boruta algorithm to rank feature importance
rf_for_boruta_cv = clone(model_cv)

boruta = BorutaPy(
    estimator=rf_for_boruta_cv,
    n_estimators="auto",
    random_state=42,
    verbose=2,
)

boruta.fit(X_training.values, y_training.values) # values because Boruta works with arrays

ranking_boruta_cv = pd.DataFrame({
    "Variable": X_training.columns,
    "Rank": boruta.ranking_,
    "Selected": boruta.support_,
    "Tentative": boruta.support_weak_
})

print("BORUTA FEATURE RANKING AFTER HYPERPARAMETER OPTIMIZATION - GRID SEARCH CV")
print(f"{ranking_boruta_cv.sort_values('Rank')}")


# --------------------------------------------------------------------------------------------
# FEATURE IMPORTANCE (MDI & BORUTA) AFTER HYPERPARAMETER OPTIMIZATION - BAYESIAN OPTIMIZATION
# --------------------------------------------------------------------------------------------
# Feature importance after hyperparameter optimization
model_bo = RandomForestRegressor(
    **best_iperparameters_bayesian_optimization,
    random_state = 42
)

model_bo.fit(X_training, y_training)

# Variable importance according to MDI (Mean Decrease Impurity)
importance_mdi_bo = pd.DataFrame({
    "Variable": X_training.columns,
    "Importance": model_bo.feature_importances_
})

importance_mdi_bo = importance_mdi_bo.sort_values(
    by="Importance",
    ascending=False
)

print("FEATURE IMPORTANCE AFTER HYPERPARAMETER OPTIMIZATION - BAYESIAN OPTIMIZATION:")
print(f"Feature importance after bayesian optimization: {importance_mdi_bo}")

# Build the Boruta algorithm to rank feature importance
rf_for_boruta_bo = clone(model_bo)

boruta = BorutaPy(
    estimator=rf_for_boruta_bo,
    n_estimators="auto",
    random_state=42,
    verbose=2,
)

boruta.fit(X_training.values, y_training.values) # values because Boruta works with arrays

ranking_boruta_bo = pd.DataFrame({
    "Variable": X_training.columns,
    "Rank": boruta.ranking_,
    "Selected": boruta.support_,
    "Tentative": boruta.support_weak_
})

print("BORUTA FEATURE RANKING AFTER HYPERPARAMETER OPTIMIZATION - BAYESIAN OPTIMIZATION")
print(f"{ranking_boruta_bo.sort_values('Rank')}")


# ---------------------------------------------------------------------
# FEATURE SELECTION THROUGH BACKWARD FEATURE SELECTION (GUIDED BY MDI)
# ---------------------------------------------------------------------
# Feature selection from MDI after Grid Search CV
ranked_features_cv = importance_mdi_cv["Variable"].tolist()
feature_sizes_cv = list(range(17, 4, -1)) # 17 = least important feature

results_features_cv = []

for n in feature_sizes_cv:
    # keep the n best features according to MDI
    selected_features_cv = ranked_features_cv[:n]

    X_training_selected = X_training[selected_features_cv]
    X_validation_selected = X_validation[selected_features_cv]

    model_sel_cv = RandomForestRegressor(
        **best_iperparameters_cross_validation,
        random_state=42
    )

    model_sel_cv.fit(X_training_selected, y_training)

    y_pred_sel = model_sel_cv.predict(X_validation_selected)

    rmse = np.sqrt(mean_squared_error(y_validation, y_pred_sel))

    mae = mean_absolute_error(y_validation, y_pred_sel)

    r2 = r2_score(y_validation, y_pred_sel)

    results_features_cv.append({
        "Number_features": n,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Features": selected_features_cv
    })

results_features_cv = pd.DataFrame(results_features_cv)

print("FEATURES SELECTED AFTER GRID SEARCH CV OPTIMIZATION")
print(results_features_cv[[
    "Number_features",
    "RMSE",
    "MAE",
    "R2"
]])

best_features_cv = results_features_cv.loc[
    results_features_cv["RMSE"].idxmin(),
    "Features"
]

print(best_features_cv)

# Feature selection from MDI after Bayesian Optimization
ranked_features_bo = importance_mdi_bo["Variable"].tolist()
feature_sizes_bo = list(range(17, 4, -1))

results_features_bo = []

for n in feature_sizes_bo:
    # keep the n best features according to MDI
    selected_features_bo = ranked_features_bo[:n]

    X_training_selected = X_training[selected_features_bo]
    X_validation_selected = X_validation[selected_features_bo]

    model_sel = RandomForestRegressor(
        **best_iperparameters_bayesian_optimization,
        random_state=42
    )

    model_sel.fit(X_training_selected, y_training)

    y_pred_sel = model_sel.predict(X_validation_selected)

    rmse = np.sqrt(mean_squared_error(y_validation, y_pred_sel))

    mae = mean_absolute_error(y_validation, y_pred_sel)

    r2 = r2_score(y_validation, y_pred_sel)

    results_features_bo.append({
        "Number_features": n,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Features": selected_features_bo
    })

results_features_bo = pd.DataFrame(results_features_bo)

print("FEATURES SELECTED AFTER BAYESIAN OPTIMIZATION")
print(results_features_bo[[
    "Number_features",
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
# Concatenate the three datasets chronologically so this backtest type can be performed
X_all = pd.concat([X_training, X_validation, X_test])
y_all = pd.concat([y_training, y_validation, y_test])
target_dates_all = dataset_h.loc[X_all.index, target_date_col]

n_train = len(X_training)
n_validation = len(X_validation)
n_test = len(X_test)


# Start of the backtest loop. cv = Cross Validation
y_predicted_list_cv = []
dates_predicted_cv = []

start = time.time()

for i in range(n_test):
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"Iteration {i}/{n_test} - {elapsed:.1f} s")
    
    target_idx = i + n_train + n_validation
    
    # Only pairs X(tau), y(tau+h) whose target is already observable at the origin date target_idx are usable.
    train_end = target_idx - orizzonte + 1
    train_window_X = X_all.iloc[i:train_end][best_features_cv]
    train_window_y = y_all.iloc[i:train_end]

    X_new = X_all.iloc[[target_idx]][best_features_cv]
    
    xgboost_cv_backtest = RandomForestRegressor(
        **best_iperparameters_cross_validation,
        n_jobs=-1,
        random_state=42,
    )
    xgboost_cv_backtest.fit(train_window_X, train_window_y)

    pred_cv = xgboost_cv_backtest.predict(X_new)[0]
    y_predicted_list_cv.append(pred_cv)
    dates_predicted_cv.append(target_dates_all.iloc[target_idx])

# Align predicted y values with dates
y_predicted_backtest_cv = pd.Series(y_predicted_list_cv, index=dates_predicted_cv, name="VIX_Forecasted")
y_true_backtest_cv = pd.Series(
    y_all.iloc[n_train + n_validation:].to_numpy(),
    index=dates_predicted_cv,
    name="VIX_Reale",
)

# Shared folder where all model files save their results
output_dir = r"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\Normale\Machine_Learning"
os.makedirs(output_dir, exist_ok=True)

# Save actual and forecast values for the Grid-Search variant so Model Confidence Set and Diebold-Mariano tests can be run later
df_out_cv = pd.DataFrame({
    "Actual": y_true_backtest_cv,
    "Forecast": y_predicted_backtest_cv
})
df_out_cv.index.name = "Date"
df_out_cv.to_csv(os.path.join(output_dir, f"random_forest_gridsearch_h{orizzonte}_v2.csv"))


# -------------------------
# ERROR METRIC CALCULATION
# -------------------------
mse_cv = mean_squared_error(y_true_backtest_cv, y_predicted_backtest_cv)
mae_cv = mean_absolute_error(y_true_backtest_cv, y_predicted_backtest_cv)
mape_cv = mean_absolute_percentage_error(y_true_backtest_cv, y_predicted_backtest_cv)
r2_cv = r2_score(y_true_backtest_cv, y_predicted_backtest_cv)
qlike_cv = np.mean((y_true_backtest_cv / y_predicted_backtest_cv) - np.log(y_true_backtest_cv / y_predicted_backtest_cv) -1)

actual_direction_cv = np.sign(y_true_backtest_cv.diff())
predicted_direction_cv = np.sign(y_predicted_backtest_cv - y_true_backtest_cv.shift(1))
directional_accuracy_cv = (actual_direction_cv == predicted_direction_cv).iloc[1:].mean() * 100

print("\n--- METRICHE BACKTEST SLIDING WINDOW RANDOM FOREST GRID-SEARCH CV ---")
print(f"MSE {orizzonte}:  {mse_cv:.4f}")
print(f"MAE:  {mae_cv:.4f}")
print(f"MAPE: {mape_cv:.4f}")
print(f"R^2:  {r2_cv:.4f}")
print(f"QLIKE: {qlike_cv:.4f}")
print(f"Directional Accuracy: {directional_accuracy_cv:.2f}%")


# -----------------------------------------------------------------------
# SLIDING WINDOW BACKTEST (WALK-FORKWARD FORECAST) BAYESIAN OPTIMIZATION
# -----------------------------------------------------------------------
# Start of the backtest loop. bo = Bayesian Optimization
y_predicted_list_bo = []
dates_predicted_bo = []

start = time.time()

for i in range(n_test):
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"Iteration {i}/{n_test} - {elapsed:.1f} s")
    
    target_idx = i + n_train + n_validation
    
    # Only pairs X(tau), y(tau+h) whose target is already observable at the origin date target_idx are usable.
    train_end = target_idx - orizzonte + 1
    train_window_X = X_all.iloc[i:train_end][best_features_bo]
    train_window_y = y_all.iloc[i:train_end]

    X_new = X_all.iloc[[target_idx]][best_features_bo]
    
    xgboost_bo_backtest = RandomForestRegressor(
        **best_iperparameters_bayesian_optimization,
        n_jobs=-1,
        random_state=42,
    )
    xgboost_bo_backtest.fit(train_window_X, train_window_y)

    pred_bo = xgboost_bo_backtest.predict(X_new)[0]
    y_predicted_list_bo.append(pred_bo)
    dates_predicted_bo.append(target_dates_all.iloc[target_idx])

# Align predicted y values with dates
y_predicted_backtest_bo = pd.Series(y_predicted_list_bo, index=dates_predicted_bo, name="VIX_Forecasted")
y_true_backtest_bo = pd.Series(
    y_all.iloc[n_train + n_validation:].to_numpy(),
    index=dates_predicted_bo,
    name="VIX_Reale",
)

# Save actual and forecast values for the Bayesian Optimization variant
df_out_bo = pd.DataFrame({
    "Actual": y_true_backtest_bo,
    "Forecast": y_predicted_backtest_bo
})
df_out_bo.index.name = "Date"
df_out_bo.to_csv(os.path.join(output_dir, f"random_forest_bayesoptimization_h{orizzonte}_v2.csv"))


# -------------------------
# ERROR METRIC CALCULATION
# -------------------------
mse_bo = mean_squared_error(y_true_backtest_bo, y_predicted_backtest_bo)
mae_bo = mean_absolute_error(y_true_backtest_bo, y_predicted_backtest_bo)
mape_bo = mean_absolute_percentage_error(y_true_backtest_bo, y_predicted_backtest_bo)
r2_bo = r2_score(y_true_backtest_bo, y_predicted_backtest_bo)
qlike_bo = np.mean((y_true_backtest_bo / y_predicted_backtest_bo) - np.log(y_true_backtest_bo / y_predicted_backtest_bo) -1)

actual_direction_bo = np.sign(y_true_backtest_bo.diff())
predicted_direction_bo = np.sign(y_predicted_backtest_bo - y_true_backtest_bo.shift(1))
directional_accuracy_bo = (actual_direction_bo == predicted_direction_bo).iloc[1:].mean() * 100

print("\n--- BACKTEST METRICS FOR SLIDING WINDOW RANDOM FOREST BAYESIAN OPTIMIZATION ---")
print(f"MSE {orizzonte}: {mse_bo:.4f}")
print(f"MAE: {mae_bo:.4f}")
print(f"MAPE: {mape_bo:.4f}")
print(f"R^2: {r2_bo:.4f}")
print(f"QLIKE: {qlike_bo:.4f}")
print(f"Directional Accuracy: {directional_accuracy_bo:.2f}%")


# ----------------------------------
# PLOTS: REAL VIX vs FORECASTED VIX  
# ----------------------------------
# Plot with Grid-Search CV optimization
# Convert the index to datetime format for a readable x-axis
y_true_backtest_cv.index = pd.to_datetime(y_true_backtest_cv.index)
y_predicted_backtest_cv.index = pd.to_datetime(y_predicted_backtest_cv.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true_backtest_cv.index, y_true_backtest_cv, label="VIX Real", color="black", linewidth=1.2)
ax.plot(y_predicted_backtest_cv.index, y_predicted_backtest_cv, label="VIX Forecast (Random Forest)", color="red", linewidth=1.2, alpha=0.8)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))   # one tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))   # format like "Jan 2022"
fig.autofmt_xdate(rotation=45)                                # rotate labels to avoid overlap

output_dir_grafici = r"C:\Users\fede1\Desktop\Repo\5_Forecasts_&_Error_Metrics\Normale\Machine_Learning"
os.makedirs(output_dir_grafici, exist_ok=True)

ax.set_title(f"Actual VIX vs Forecast VIX (Test Set) with hyperparameter optimization via Grid Search CV h = {orizzonte}, v2")
ax.set_xlabel("Date")
ax.set_ylabel("VIX")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir_grafici, f"random_forest_CV_vix_reale_vs_previsto_h{orizzonte}_v2.png"), dpi=300, bbox_inches="tight")
plt.show()


# Plot with Bayesian Optimization
y_true_backtest_bo.index = pd.to_datetime(y_true_backtest_bo.index)
y_predicted_backtest_bo.index = pd.to_datetime(y_predicted_backtest_bo.index)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true_backtest_bo.index, y_true_backtest_bo, label="VIX Real", color="black", linewidth=1.2)
ax.plot(y_predicted_backtest_bo.index, y_predicted_backtest_bo, label="VIX Forecast (Random Forest)", color="red", linewidth=1.2, alpha=0.8)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))   
fig.autofmt_xdate(rotation=45)                                

ax.set_title(f"Actual VIX vs Forecast VIX (Test Set) with Bayesian Optimization h = {orizzonte}, v2")
ax.set_xlabel("Date")
ax.set_ylabel("VIX")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir_grafici, f"random_forest_BO_vix_reale_vs_previsto_h{orizzonte}_v2.png"), dpi=300, bbox_inches="tight")
plt.show()