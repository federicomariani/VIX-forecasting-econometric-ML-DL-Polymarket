import os
import time
import random
import numpy as np
import pandas as pd
import torch
import optuna
from itertools import product
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import logging
from multiprocessing import Pool



warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def torch_thread_limit(): # Evita che ogni processo provi a usare tutti i thread disponibili
    torch.set_num_threads(1)


def run_backtest_iteration(args):
    """
    Esegue una singola iterazione del backtest walk-forward.
    Riceve tutto il necessario come argomenti (niente variabili globali),
    perché su Windows ogni processo figlio parte 'pulito'.
    """
    (i, dataset_h, training_dataset, n_train, n_validation,
     max_encoder_length, max_prediction_length, hyperparams,
     epochs_backtest, median_idx) = args

    torch_thread_limit()

    train_window_df = dataset_h[(dataset_h.time_idx >= i) & (dataset_h.time_idx < i + n_train + n_validation)]

    window_training_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, train_window_df, predict=False, stop_randomization=True
    )
    window_train_dataloader = window_training_dataset.to_dataloader(
        train=True, batch_size=32, num_workers=0
    )

    forecast_origin_idx = i + n_train + n_validation - 1
    target_idx = forecast_origin_idx + max_prediction_length

    pred_slice = dataset_h[
        (dataset_h.time_idx >= forecast_origin_idx - max_encoder_length + 1)
        & (dataset_h.time_idx <= target_idx)
    ]
    
    if len(pred_slice) != max_encoder_length + max_prediction_length:
        return None

    pred_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, pred_slice, predict=True, stop_randomization=True
    )
    pred_dataloader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        window_training_dataset, loss=QuantileLoss(), optimizer="Adam", **hyperparams,
    )
    trainer = pl.Trainer(
        max_epochs=epochs_backtest,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
    )
    trainer.fit(model, train_dataloaders=window_train_dataloader)

    raw_pred = model.predict(pred_dataloader, mode="quantiles")

    # Previsione puntuale: quantile 0.50 dell'ultimo step del decoder
    pred_value = raw_pred[0, -1, median_idx].item()

    # Controllo che il target sia presente nel dataset
    if target_idx not in dataset_h.time_idx.values:
        return None

    # Data effettivamente associata alla previsione
    date = dataset_h.loc[
        dataset_h.time_idx == target_idx,
        "Date"
    ].values[0]

    return (i, pred_value, date, target_idx)

# ------------------------
# CONFIGURAZIONE GENERALE 
# ------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    pl.seed_everything(42)

    N_WORKERS = 0

    orizzonte = 1 # da cambiare manualmente ad ogni run
    LOOKBACK = 20

    MAX_EPOCHS_TUNING = 12
    EARLY_STOPPING_PATIENCE = 3
    EPOCHS_BACKTEST = 12

    RUN_FULL_BACKTEST = True # se impostato su "False", fa il backtest solo su 10 osservazioni
    TEST_SUBSET = 10

    N_PROCESSES = 8 # iterazioni di prova per stimare i tempi


    # ---------------
    # IMPORT DATASET
    # ---------------
    dataset = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\3. Deep Learning\0. Dataset\Data\Clean\dataset.csv")
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset = dataset.sort_values("Date").reset_index(drop=True)


    # -----------------
    # DATA PREPARATION
    # -----------------
    predittori = ["Log_Return_S&P500", "Log_Return_WTI", "Close_OVX", "OVX_VIX_spread",
                "VIX_lag1", "VIX_lag5", "VIX_lag22", "RSI_14d", "Log_Return_DXY",
                "Yield_curve", "Log_difference_initial_claims", "VIX_MA5", "VIX_MA10", "VIX_MA20",
                "Credit_Spread", "Log_Return_Gold", "Weekday"]

    # Come previsione puntuale si prende il quantile 0.5 dell'ultimo step decoder (giorno h)
    dataset_h = dataset.dropna(subset=predittori + ["Close_VIX"]).reset_index(drop=True)
    dataset_h["time_idx"] = np.arange(len(dataset_h))
    dataset_h["group"] = "VIX"

    mask_training = dataset_h["Date"] <= "2019-12-31"
    mask_validation = (dataset_h["Date"] >= "2020-01-01") & (dataset_h["Date"] <= "2021-12-31")
    mask_test = dataset_h["Date"] >= "2022-01-01"

    n_train = int(mask_training.sum())
    n_validation = int(mask_validation.sum())
    n_test = int(mask_test.sum())

    training_cutoff_idx = dataset_h.loc[mask_training, "time_idx"].max()
    validation_cutoff_idx = dataset_h.loc[mask_validation, "time_idx"].max()

    print(f"n_train={n_train}, n_validation={n_validation}, n_test={n_test}")


    # -------------------------------------------------------------------------
    # TIMESERIESDATASET DI TRAINING E VALIDATION (per tutta la fase di tuning)
    # -------------------------------------------------------------------------
    max_prediction_length = orizzonte
    max_encoder_length = LOOKBACK

    time_varying_unknown_reals = [p for p in predittori if p != "Weekday"] + ["Close_VIX"]

    training_dataset = TimeSeriesDataSet(
        dataset_h[dataset_h.time_idx <= training_cutoff_idx],
        time_idx="time_idx",
        target="Close_VIX",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "Weekday"],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Finestre di validation: partiamo da training_cutoff_idx - max_encoder_length + 1 così tutte le finestre generate hanno il target dentro il periodo di validation
    validation_slice = dataset_h[
        (dataset_h.time_idx >= training_cutoff_idx - max_encoder_length + 1)
        & (dataset_h.time_idx <= validation_cutoff_idx)
    ]
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, validation_slice, predict=False, stop_randomization=True
    )

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=32, num_workers=N_WORKERS)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=32, num_workers=N_WORKERS)

    # Quantili usati dalla QuantileLoss (di default) e indice del 50esimo percentile
    quantili = QuantileLoss().quantiles
    median_idx = quantili.index(0.5)


    # ---------------------------------------------
    # MODELLO BASE TFT + VARIABLE IMPORTANCE (VSN)
    # ---------------------------------------------
    tft_baseline = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="Adam",
    )

    trainer_baseline = pl.Trainer(
        max_epochs=MAX_EPOCHS_TUNING,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min")],
    )
    trainer_baseline.fit(tft_baseline, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    raw_predictions_baseline = tft_baseline.predict(val_dataloader, mode="raw", return_x=True)
    interpretation_baseline = tft_baseline.interpret_output(raw_predictions_baseline.output, reduction="sum")

    ranking = pd.DataFrame({
        "Variabile": tft_baseline.encoder_variables,
        "Importanza": interpretation_baseline["encoder_variables"].numpy(),
    })
    ranking = ranking[ranking["Variabile"].isin(predittori)]
    ranking = ranking.sort_values("Importanza", ascending=False).reset_index(drop=True)

    print("RANKING VARIABLE IMPORTANCE (TFT - Variable Selection Network)")
    print(ranking)


    # ---------------
    # GRID SEARCH CV 
    # ---------------
    param_grid_cross_validation = {
        "hidden_size": [8, 16, 24],
        "hidden_continuous_size": [8, 16],
        "attention_head_size": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.01, 0.03, 0.1],
    }

    combinazioni_grid = [
        dict(zip(param_grid_cross_validation.keys(), values))
        for values in product(*param_grid_cross_validation.values())
    ]
    
    n_iter_grid_search = len(combinazioni_grid)
    
    start_cross_validation = time.time()
    risultati_grid = []

    for params in combinazioni_grid:
        pl.seed_everything(42)

        tft_grid = TemporalFusionTransformer.from_dataset(
            training_dataset, loss=QuantileLoss(), optimizer="Adam", **params,
        )
        trainer_grid = pl.Trainer(
            max_epochs=MAX_EPOCHS_TUNING,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=0.1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min")],
        )
        trainer_grid.fit(tft_grid, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        val_loss = trainer_grid.callback_metrics["val_loss"].item()
        risultati_grid.append({**params, "val_loss": val_loss})
        print(f"Grid combo {params} -> val_loss={val_loss:.4f}")

    tempo_grid_search = time.time() - start_cross_validation
    risultati_grid_df = pd.DataFrame(risultati_grid)

    best_iperparameters_cross_validation = risultati_grid_df.loc[risultati_grid_df["val_loss"].idxmin()].drop("val_loss").to_dict()
    best_iperparameters_cross_validation["hidden_size"] = int(best_iperparameters_cross_validation["hidden_size"])
    best_iperparameters_cross_validation["attention_head_size"] = int(best_iperparameters_cross_validation["attention_head_size"])
    best_iperparameters_cross_validation["hidden_continuous_size"] = int(best_iperparameters_cross_validation["hidden_continuous_size"])

    print("MIGLIORI IPERPARAMETRI GRID-SEARCH CV:", best_iperparameters_cross_validation)
    print("TEMPO DI TUNING GRID-SEARCH CV:", tempo_grid_search)


    # -------------------------------------
    # BAYESIAN OPTIMIZATION (Optuna / TPE)
    # -------------------------------------
    search_spaces_bayesian_optimization = {
        "hidden_size": [8, 16, 24],
        "hidden_continuous_size": [8, 16],
        "attention_head_size": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.01, 0.03, 0.1],
    }

    start_bayesian_optimization = time.time()

    def objective(trial):
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in search_spaces_bayesian_optimization.items()
        }

        pl.seed_everything(42)

        tft_bayes = TemporalFusionTransformer.from_dataset(
            training_dataset,
            loss=QuantileLoss(),
            optimizer="Adam",
            **params,
        )

        trainer_bayes = pl.Trainer(
            max_epochs=MAX_EPOCHS_TUNING,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOPPING_PATIENCE,
                    mode="min",
                )
            ],
        )

        trainer_bayes.fit(
            tft_bayes,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        return trainer_bayes.callback_metrics["val_loss"].item()


    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials = n_iter_grid_search)

    tempo_bayesian_optimization = time.time() - start_bayesian_optimization
    best_iperparameters_bayesian_optimization = dict(study.best_trial.params)

    print("MIGLIORI IPERPARAMETRI BAYESIAN OPTIMIZATION:", best_iperparameters_bayesian_optimization)
    print("TEMPO DI TUNING BAYESIAN OPTIMIZATION:", tempo_bayesian_optimization)


    # -----------------------------------------------------------------------
    # FEATURES IMPORTANCE DOPO OTTIMIZZAZIONE IPERPARAMETRI - GRID SEARCH CV
    # -----------------------------------------------------------------------
    tft_cv = TemporalFusionTransformer.from_dataset(
        training_dataset, loss=QuantileLoss(), optimizer="Adam", **best_iperparameters_cross_validation,
    )
    trainer_cv = pl.Trainer(
        max_epochs=MAX_EPOCHS_TUNING,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min")],
    )
    trainer_cv.fit(tft_cv, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    raw_pred_cv = tft_cv.predict(val_dataloader, mode="raw", return_x=True)
    interpretation_cv = tft_cv.interpret_output(raw_pred_cv.output, reduction="sum")
    importance_cv = pd.DataFrame({
        "Variabile": tft_cv.encoder_variables,
        "Importanza": interpretation_cv["encoder_variables"].numpy(),
    })
    importance_cv = importance_cv[importance_cv["Variabile"].isin(predittori)]
    importance_cv = importance_cv.sort_values("Importanza", ascending=False).reset_index(drop=True)

    print("IMPORTANZA FEATURES DOPO OTTIMIZZAZIONE IPERPARAMETRI - GRID SEARCH CV:")
    print(importance_cv)


    # ------------------------------------------------------------------------------
    # FEATURES IMPORTANCE DOPO OTTIMIZZAZIONE IPERPARAMETRI - BAYESIAN OPTIMIZATION
    # ------------------------------------------------------------------------------
    tft_bo = TemporalFusionTransformer.from_dataset(
        training_dataset, loss=QuantileLoss(), optimizer="Adam", **best_iperparameters_bayesian_optimization,
    )
    trainer_bo = pl.Trainer(
        max_epochs=MAX_EPOCHS_TUNING,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min")],
    )
    trainer_bo.fit(tft_bo, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    raw_pred_bo = tft_bo.predict(val_dataloader, mode="raw", return_x=True)
    interpretation_bo = tft_bo.interpret_output(raw_pred_bo.output, reduction="sum")
    importance_bo = pd.DataFrame({
        "Variabile": tft_bo.encoder_variables,
        "Importanza": interpretation_bo["encoder_variables"].numpy(),
    })
    importance_bo = importance_bo[importance_bo["Variabile"].isin(predittori)]
    importance_bo = importance_bo.sort_values("Importanza", ascending=False).reset_index(drop=True)

    print("IMPORTANZA FEATURES DOPO OTTIMIZZAZIONE IPERPARAMETRI - BAYESIAN OPTIMIZATION:")
    print(importance_bo)


    # -----------------------------------------------------------------------------
    # BACKWARD FEATURE SELECTION (guidata da Variable Importance) - GRID SEARCH CV
    # -----------------------------------------------------------------------------
    ranked_features_cv = importance_cv["Variabile"].tolist()
    feature_sizes_cv = list(range(17, 4, -1))

    results_features_cv = []

    for n in feature_sizes_cv:
        selected_features_cv = ranked_features_cv[:n]
        unknown_reals_sel = [f for f in selected_features_cv if f != "Weekday"] + ["Close_VIX"]

        training_sel = TimeSeriesDataSet(
            dataset_h[dataset_h.time_idx <= training_cutoff_idx],
            time_idx="time_idx",
            target="Close_VIX",
            group_ids=["group"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=["time_idx"] + (["Weekday"] if "Weekday" in selected_features_cv else []),
            time_varying_unknown_reals=unknown_reals_sel,
            target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        validation_sel = TimeSeriesDataSet.from_dataset(training_sel, validation_slice, predict=False, stop_randomization=True)

        dl_train_sel = training_sel.to_dataloader(train=True, batch_size=32, num_workers=N_WORKERS)
        dl_val_sel = validation_sel.to_dataloader(train=False, batch_size=32, num_workers=N_WORKERS)

        pl.seed_everything(42)

        model_sel_cv = TemporalFusionTransformer.from_dataset(
            training_sel, loss=QuantileLoss(), optimizer="Adam", **best_iperparameters_cross_validation,
        )
        trainer_sel = pl.Trainer(
            max_epochs=MAX_EPOCHS_TUNING,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=0.1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min")],
        )
        trainer_sel.fit(model_sel_cv, train_dataloaders=dl_train_sel, val_dataloaders=dl_val_sel)

        preds_sel = model_sel_cv.predict(dl_val_sel, mode="quantiles")
        y_pred_sel = preds_sel[:, -1, median_idx].numpy()

        y_true_sel = []
        for x_batch, y_batch in iter(dl_val_sel):
            y_true_sel.append(y_batch[0][:, -1].numpy())
        y_true_sel = np.concatenate(y_true_sel)

        rmse = np.sqrt(mean_squared_error(y_true_sel, y_pred_sel))
        mae = mean_absolute_error(y_true_sel, y_pred_sel)
        r2 = r2_score(y_true_sel, y_pred_sel)

        results_features_cv.append({
            "Numero_feature": n, "RMSE": rmse, "MAE": mae, "R2": r2, "Features": selected_features_cv
        })
        print(f"[GRID] n={n} -> RMSE={rmse:.4f}")

    results_features_cv = pd.DataFrame(results_features_cv)
    print("FEATURES SELEZIONATE DOPO OTTIMIZZAZIONE GRID SEARCH CV")
    print(results_features_cv[["Numero_feature", "RMSE", "MAE", "R2"]])

    best_features_cv = results_features_cv.loc[results_features_cv["RMSE"].idxmin(), "Features"]
    print(best_features_cv)
    
    unknown_reals_cv_final = [
        f for f in best_features_cv if f != "Weekday"
    ] + ["Close_VIX"]

    training_dataset_cv_final = TimeSeriesDataSet(
        dataset_h[dataset_h.time_idx <= validation_cutoff_idx],
        time_idx="time_idx",
        target="Close_VIX",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"] +
            (["Weekday"] if "Weekday" in best_features_cv else []),
        time_varying_unknown_reals=unknown_reals_cv_final,
        target_normalizer=GroupNormalizer(
            groups=["group"],
            transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


    # ------------------------------------------------------------------------------------
    # BACKWARD FEATURE SELECTION (guidata da Variable Importance) - BAYESIAN OPTIMIZATION
    # ------------------------------------------------------------------------------------
    ranked_features_bo = importance_bo["Variabile"].tolist()
    feature_sizes_bo = list(range(17, 4, -1))

    results_features_bo = []

    for n in feature_sizes_bo:
        selected_features_bo = ranked_features_bo[:n]
        unknown_reals_sel = [f for f in selected_features_bo if f != "Weekday"] + ["Close_VIX"]

        training_sel = TimeSeriesDataSet(
            dataset_h[dataset_h.time_idx <= training_cutoff_idx],
            time_idx="time_idx",
            target="Close_VIX",
            group_ids=["group"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=["time_idx"] + (["Weekday"] if "Weekday" in selected_features_bo else []),
            time_varying_unknown_reals=unknown_reals_sel,
            target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        validation_sel = TimeSeriesDataSet.from_dataset(training_sel, validation_slice, predict=False, stop_randomization=True)

        dl_train_sel = training_sel.to_dataloader(train=True, batch_size=32, num_workers=N_WORKERS)
        dl_val_sel = validation_sel.to_dataloader(train=False, batch_size=32, num_workers=N_WORKERS)

        pl.seed_everything(42)

        model_sel_bo = TemporalFusionTransformer.from_dataset(
            training_sel, loss=QuantileLoss(), optimizer="Adam", **best_iperparameters_bayesian_optimization,
        )
        trainer_sel = pl.Trainer(
            max_epochs=8,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=0.1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min")],
        )
        trainer_sel.fit(model_sel_bo, train_dataloaders=dl_train_sel, val_dataloaders=dl_val_sel)

        preds_sel = model_sel_bo.predict(dl_val_sel, mode="quantiles")
        y_pred_sel = preds_sel[:, -1, median_idx].numpy()

        y_true_sel = []
        for x_batch, y_batch in iter(dl_val_sel):
            y_true_sel.append(y_batch[0][:, -1].numpy())
        y_true_sel = np.concatenate(y_true_sel)

        rmse = np.sqrt(mean_squared_error(y_true_sel, y_pred_sel))
        mae = mean_absolute_error(y_true_sel, y_pred_sel)
        r2 = r2_score(y_true_sel, y_pred_sel)

        results_features_bo.append({
            "Numero_feature": n, "RMSE": rmse, "MAE": mae, "R2": r2, "Features": selected_features_bo
        })
        print(f"[BAYES] n={n} -> RMSE={rmse:.4f}")

    results_features_bo = pd.DataFrame(results_features_bo)
    print("FEATURES SELEZIONATE DOPO BAYESIAN OPTIMIZATION")
    print(results_features_bo[["Numero_feature", "RMSE", "MAE", "R2"]])

    best_features_bo = results_features_bo.loc[results_features_bo["RMSE"].idxmin(), "Features"]
    print(best_features_bo)

    unknown_reals_bo_final = [
        f for f in best_features_bo if f != "Weekday"
    ] + ["Close_VIX"]

    training_dataset_bo_final = TimeSeriesDataSet(
        dataset_h[dataset_h.time_idx <= validation_cutoff_idx],
        time_idx="time_idx",
        target="Close_VIX",
        group_ids=["group"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"] +
            (["Weekday"] if "Weekday" in best_features_bo else []),
        time_varying_unknown_reals=unknown_reals_bo_final,
        target_normalizer=GroupNormalizer(
            groups=["group"],
            transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # --------------------------------------------------------
    # SLIDING WINDOW BACKTEST (WALK-FORWARD) - GRID SEARCH CV
    # --------------------------------------------------------
    n_iterazioni = (
        TEST_SUBSET if not RUN_FULL_BACKTEST
        else n_test - max_prediction_length + 1
    )
    
    if not RUN_FULL_BACKTEST:
        print(f"\n*** MODALITA' PROVA: eseguo solo {TEST_SUBSET} iterazioni su {n_test} per stimare i tempi. ***")
        print("*** Imposta RUN_FULL_BACKTEST = True per eseguire il backtest completo. ***\n")

    start = time.time()

    args_list_cv = [
        (i, dataset_h, training_dataset_cv_final, n_train, n_validation,
        max_encoder_length, max_prediction_length,
        best_iperparameters_cross_validation,
        EPOCHS_BACKTEST, median_idx)
        for i in range(n_iterazioni)
    ]

    print(f"Avvio backtest GRID SEARCH CV parallelizzato su {N_PROCESSES} processi...")

    risultati_cv = []
    n_completate_cv = 0

    with Pool(processes=N_PROCESSES) as pool:
        for risultato in pool.imap_unordered(run_backtest_iteration, args_list_cv):
            risultati_cv.append(risultato)
            n_completate_cv += 1
            if n_completate_cv % 10 == 0 or n_completate_cv == len(args_list_cv):
                elapsed = time.time() - start
                print(f"Iterazione {n_completate_cv}/{n_iterazioni} - {elapsed:.1f} s")

    risultati_cv = [r for r in risultati_cv if r is not None]
    risultati_cv.sort(key=lambda r: r[0])  # riordina per indice i, l'ordine può mescolarsi in parallelo

    y_predicted_list_cv = [r[1] for r in risultati_cv]
    dates_predicted_cv = [r[2] for r in risultati_cv]
    targets_predicted_cv = [r[3] for r in risultati_cv]

    tempo_backtest_cv = time.time() - start
    print(f"Backtest GRID SEARCH CV completato in {tempo_backtest_cv:.1f} s "
          f"({len(y_predicted_list_cv)} iterazioni valide su {n_iterazioni})")

    y_predicted_backtest_cv = pd.Series(y_predicted_list_cv, index=pd.to_datetime(dates_predicted_cv), name="VIX_Forecasted")
    y_true_backtest_cv = dataset_h.set_index("time_idx").loc[
        targets_predicted_cv, "Close_VIX"
    ]   
    y_true_backtest_cv.index = y_predicted_backtest_cv.index
    y_true_backtest_cv = y_true_backtest_cv.rename("VIX_Reale")

    output_dir = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\3. Deep Learning\2. TFT\Results\0_Forecast"
    os.makedirs(output_dir, exist_ok=True)

    df_out_cv = pd.DataFrame({"y_true": y_true_backtest_cv, "y_pred": y_predicted_backtest_cv})
    df_out_cv.to_csv(os.path.join(output_dir, f"tft_gridsearch_h{orizzonte}.csv"))


    # ------------------------------------------
    # CALCOLO METRICHE ERRORE - GRID SEARCH CV
    # ------------------------------------------
    mse_cv = mean_squared_error(y_true_backtest_cv, y_predicted_backtest_cv)
    mae_cv = mean_absolute_error(y_true_backtest_cv, y_predicted_backtest_cv)
    mape_cv = mean_absolute_percentage_error(y_true_backtest_cv, y_predicted_backtest_cv)
    r2_cv = r2_score(y_true_backtest_cv, y_predicted_backtest_cv)
    qlike_cv = np.mean((y_true_backtest_cv / y_predicted_backtest_cv) - np.log(y_true_backtest_cv / y_predicted_backtest_cv) - 1)

    # Directional Accuracy CV
    # Valore reale al tempo t
    y_true_t_cv = y_true_backtest_cv.shift(orizzonte)

    # Direzione effettivamente realizzata: VIX(t+h) - VIX(t)
    actual_direction_cv = np.sign(
        y_true_backtest_cv - y_true_t_cv
    )

    # Direzione prevista: VIX_hat(t+h) - VIX(t)
    predicted_direction_cv = np.sign(
        y_predicted_backtest_cv - y_true_t_cv
    )

    # Elimina le osservazioni senza valore di riferimento
    mask_direction_cv = (
        y_true_t_cv.notna()
        & actual_direction_cv.notna()
        & predicted_direction_cv.notna()
    )

    directional_accuracy_cv = (
        actual_direction_cv[mask_direction_cv]
        == predicted_direction_cv[mask_direction_cv]
    ).mean() * 100

    print("\n--- METRICHE BACKTEST SLIDING WINDOW TFT GRID-SEARCH CV ---")
    print(f"MSE {orizzonte}:  {mse_cv:.4f}")
    print(f"MAE:  {mae_cv:.4f}")
    print(f"MAPE: {mape_cv:.4f}")
    print(f"R^2:  {r2_cv:.4f}")
    print(f"QLIKE: {qlike_cv:.4f}")
    print(f"Directional Accuracy: {directional_accuracy_cv:.2f}%")


    # ---------------------------------------------------------------
    # SLIDING WINDOW BACKTEST (WALK-FORWARD) - BAYESIAN OPTIMIZATION
    # ---------------------------------------------------------------
    y_predicted_list_bo = []
    dates_predicted_bo = []
    targets_predicted_bo = []

    start = time.time()

    args_list_bo = [
        (i, dataset_h, training_dataset_bo_final, n_train, n_validation,
        max_encoder_length, max_prediction_length,
        best_iperparameters_bayesian_optimization,
        EPOCHS_BACKTEST, median_idx)
        for i in range(n_iterazioni)
    ]

    print(f"Avvio backtest BAYESIAN OPTIMIZATION parallelizzato su {N_PROCESSES} processi...")

    risultati_bo = []
    n_completate_bo = 0

    with Pool(processes=N_PROCESSES) as pool:
        for risultato in pool.imap_unordered(run_backtest_iteration, args_list_bo):
            risultati_bo.append(risultato)
            n_completate_bo += 1
            if n_completate_bo % 10 == 0 or n_completate_bo == len(args_list_bo):
                elapsed = time.time() - start
                print(f"Iterazione {n_completate_bo}/{n_iterazioni} - {elapsed:.1f} s")

    risultati_bo = [r for r in risultati_bo if r is not None]
    risultati_bo.sort(key=lambda r: r[0])

    y_predicted_list_bo = [r[1] for r in risultati_bo]
    dates_predicted_bo = [r[2] for r in risultati_bo]
    targets_predicted_bo = [r[3] for r in risultati_bo]

    tempo_backtest_bo = time.time() - start
    print(f"Backtest BAYESIAN OPTIMIZATION completato in {tempo_backtest_bo:.1f} s "
          f"({len(y_predicted_list_bo)} iterazioni valide su {n_iterazioni})")

    y_predicted_backtest_bo = pd.Series(y_predicted_list_bo, index=pd.to_datetime(dates_predicted_bo), name="VIX_Forecasted")
    y_true_backtest_bo = dataset_h.set_index("time_idx").loc[
        targets_predicted_bo, "Close_VIX"
    ]
    y_true_backtest_bo.index = y_predicted_backtest_bo.index
    y_true_backtest_bo = y_true_backtest_bo.rename("VIX_Reale")

    df_out_bo = pd.DataFrame({"y_true": y_true_backtest_bo, "y_pred": y_predicted_backtest_bo})
    df_out_bo.to_csv(os.path.join(output_dir, f"tft_bayesoptimization_h{orizzonte}.csv"))


    # ------------------------------------------------
    # CALCOLO METRICHE ERRORE - BAYESIAN OPTIMIZATION
    # ------------------------------------------------
    mse_bo = mean_squared_error(y_true_backtest_bo, y_predicted_backtest_bo)
    mae_bo = mean_absolute_error(y_true_backtest_bo, y_predicted_backtest_bo)
    mape_bo = mean_absolute_percentage_error(y_true_backtest_bo, y_predicted_backtest_bo)
    r2_bo = r2_score(y_true_backtest_bo, y_predicted_backtest_bo)
    qlike_bo = np.mean((y_true_backtest_bo / y_predicted_backtest_bo) - np.log(y_true_backtest_bo / y_predicted_backtest_bo) - 1)

    # Directional Accuracy BO
    # Valore reale al tempo t
    y_true_t_bo = y_true_backtest_bo.shift(orizzonte)

    # Direzione effettivamente realizzata: VIX(t+h) - VIX(t)
    actual_direction_bo = np.sign(
        y_true_backtest_bo - y_true_t_bo
    )

    # Direzione prevista: VIX_hat(t+h) - VIX(t)
    predicted_direction_bo = np.sign(
        y_predicted_backtest_bo - y_true_t_bo
    )

    # Elimina le osservazioni senza valore di riferimento
    mask_direction_bo = (
        y_true_t_bo.notna()
        & actual_direction_bo.notna()
        & predicted_direction_bo.notna()
    )

    directional_accuracy_bo = (
        actual_direction_bo[mask_direction_bo]
        == predicted_direction_bo[mask_direction_bo]
    ).mean() * 100

    print("\n--- METRICHE BACKTEST SLIDING WINDOW TFT BAYESIAN OPTIMIZATION ---")
    print(f"MSE {orizzonte}: {mse_bo:.4f}")
    print(f"MAE: {mae_bo:.4f}")
    print(f"MAPE: {mape_bo:.4f}")
    print(f"R^2: {r2_bo:.4f}")
    print(f"QLIKE: {qlike_bo:.4f}")
    print(f"Directional Accuracy: {directional_accuracy_bo:.2f}%")


    # -------------------------------------
    # GRAFICI: VIX REALE vs VIX FORECASTED
    # -------------------------------------
    output_dir_grafici = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\3. Deep Learning\2. TFT\Results\1_Grafici_backtest"
    os.makedirs(output_dir_grafici, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_true_backtest_cv.index, y_true_backtest_cv, label="VIX Reale", color="black", linewidth=1.2)
    ax.plot(y_predicted_backtest_cv.index, y_predicted_backtest_cv, label="VIX Previsto (TFT)", color="red", linewidth=1.2, alpha=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)
    ax.set_title(f"VIX Reale vs VIX Previsto (Test Set) con ottimizzazione iperparametri tramite Grid Search CV h = {orizzonte}")
    ax.set_xlabel("Data")
    ax.set_ylabel("VIX")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_grafici, f"tft_CV_vix_reale_vs_previsto_h{orizzonte}.png"), dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_true_backtest_bo.index, y_true_backtest_bo, label="VIX Reale", color="black", linewidth=1.2)
    ax.plot(y_predicted_backtest_bo.index, y_predicted_backtest_bo, label="VIX Previsto (TFT)", color="red", linewidth=1.2, alpha=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)
    ax.set_title(f"VIX Reale vs VIX Previsto (Test Set) con Bayesian Optimization h = {orizzonte}")
    ax.set_xlabel("Data")
    ax.set_ylabel("VIX")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_grafici, f"tft_BO_vix_reale_vs_previsto_h{orizzonte}.png"), dpi=300, bbox_inches="tight")
    plt.show()