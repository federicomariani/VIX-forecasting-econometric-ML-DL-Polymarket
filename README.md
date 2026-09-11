# Previsione del VIX tra modelli econometrici, Machine Learning e Deep Learning: un confronto multi-orizzonte con l'integrazione di dati da Polymarket

Tesi di Laurea Triennale in Economia: Banche, Aziende e Mercati

**Università degli Studi di Macerata** - Dipartimento di Economia e Diritto
Anno Accademico 2025-2026

- **Candidato:** Federico Mariani
- **Relatore:** Prof. Luca Romeo
- **Correlatore:** Prof. Andrea Bucci

---

## Abstract

Il lavoro confronta modelli di previsione del CBOE Volatility Index (VIX) appartenenti a quattro famiglie:

- **Random Walk** (benchmark)
- **Econometrici** (ARIMA(1,1,1), HAR-Type)
- **Machine Learning** (Random Forest, XGBoost)
- **Deep Learning** (LSTM, Temporal Fusion Transformer)

su tre orizzonti temporali (**1, 5 e 22 giorni**), utilizzando lo stesso arco temporale per tutti i modelli.

Un contributo originale della tesi è l'integrazione di feature derivanti da mercati predittivi (**Polymarket** - decisioni FOMC sui tassi, probabilità di recessione USA, inflazione CPI YoY) nei modelli econometrici, di Machine Learning e Deep Learning, per valutare se le probabilità implicite in questi mercati aggiungano informazione utile alla previsione. Per ogni modello viene quindi prodotta una coppia di run **"con PM"** / **"senza PM"**, confrontate tra loro.

I modelli sono ottimizzati sia con **Grid Search** sia con **Bayesian Optimization**, e confrontati tramite metriche di accuratezza puntuale (MSE, MAE, MAPE, R², QLIKE), Directional Accuracy, test di **Diebold-Mariano** e **Model Confidence Set (MCS)**, anche su diversi regimi di volatilità (Low / Medium / High).

---

## Struttura della repository

```
├── 0_Dataset/
│   ├── Data/
│   │   ├── Raw/                         # Serie macro grezze da FRED (CSV)
│   │   │   ├── credit_spread.csv
│   │   │   ├── initial_claims.csv
│   │   │   ├── tbill_3m.csv
│   │   │   └── tyield_10y.csv
│   │   ├── Clean/
│   │   │   ├── dataset.csv                       # dataset finale usato dai modelli senza Polymarket
│   │   │   ├── dataset_adj_x_polymarket.csv       # stesso dataset finale, allineato in lunghezza a Polymarket
│   │   │   └── Info.txt
│   │   └── download_data.py             # scarica VIX + regressori (yfinance, FRED) e crea le feature
│   │
│   └── Polymarket_data/
│       ├── Raw/                         # dati grezzi scaricati dalle API di Polymarket, per evento
│       │   ├── 0_FOMC_Interest_Rates/   # una sottocartella per ogni riunione FOMC (2025-2026)
│       │   ├── 1_US_Recession/          # mercati "recessione USA" 2025 / 2026
│       │   └── 2_CPI_YoY/               # mercati inflazione CPI YoY, mese per mese
│       ├── Clean/                       # stessi dati puliti + df_polymarket.csv aggregato
│       ├── dataset_w_polymarket.csv     # dataset finale utilizzato dai modelli con Polymarket: feature tradizionali + feature Polymarket
│       ├── download_data_polymarket.py  # scarica dati via API pubbliche di Polymarket
│       ├── clean_polymarket.py          # pulizia dati (step 1)
│       ├── clean_polymarket_2.py        # pulizia dati (step 2)
│       ├── feature_engineering_polymarket.py  # costruzione feature da probabilità implicite
│       └── Info.txt
│
├── 1_Random_Walk/
│   └── random_walk.py                   # modello benchmark
│
├── 2_Econometrics/
│   ├── 1_ARIMA(1,1,1)/
│   │   ├── arima.py                     # senza feature Polymarket
│   │   └── arima_pm.py                  # con feature Polymarket
│   └── 2_HAR-Type/
│       ├── har_type.py
│       └── har_type_pm.py
│
├── 3_Machine_Learning/
│   ├── 1_Random_Forest/
│   │   ├── random_forest.py             # lag VIX a 1, 5, 22 giorni
│   │   ├── random_forest_pm.py          # variante con feature Polymarket
│   │   └── random_forest_v2.py          # variante con tutti i lag VIX fino a 22 giorni
│   └── 2_XGBoost/
│       ├── xgboost_model.py
│       ├── xgboost_pm.py
│       └── xgboost_v2.py
│
├── 4_Deep_Learning/
│   ├── 1_LSTM/
│   │   ├── lstm.py
│   │   └── lstm_pm.py
│   └── 2_TFT/
│       ├── tft.py                       # Temporal Fusion Transformer (pytorch-forecasting)
│       └── tft_pm.py
│
├── 5_Forecasts_&_Error_Metrics/         # previsioni, grafici e risultati di tuning per ciascun modello
│   ├── Normale/                         # run senza feature Polymarket
│   │   ├── Modelli_econometrici/{h1,h5,h22}/
│   │   ├── Machine_Learning/{h1,h5,h22}/[v2/]
│   │   └── Deep_Learning/{h1,h5,h22}/
│   └── With_Polymarket/                 # run con feature Polymarket, "con_PM" vs "senza_PM", stesso schema di cartelle
│       ├── Modelli_econometrici/{h1,h5,h22}/
│       ├── Machine_Learning/{h1,h5,h22}/
│       └── Deep_Learning/{h1,h5,h22}/
│
└── 6_Final_Results/                     # valutazione comparativa finale tra tutti i modelli
    ├── metriche.py                      # calcolo MSE, MAE, MAPE, R², QLIKE, Directional Accuracy
    ├── regimi_vol.py                    # analisi per regime di volatilità (Low/Medium/High)
    ├── mcs.py                           # Model Confidence Set
    ├── Results/                         # output: stats_models_h*.png, mcs_h*.png, vol_regime_h*.csv
    └── With_Polymarket/
        ├── metriche_pm.py
        ├── regimi_vol_pm.py
        ├── mcs_pm.py
        ├── diebold_mariano.py           # test di Diebold-Mariano (con PM vs senza PM)
        └── Results/                     # dm_h*.png, mcs_mse_h*.png, mcs_qlike_h*.png, stats_models_pm_h*.png
```

Ogni script `*_pm.py` è la controparte di quello omonimo, addestrata includendo anche le feature Polymarket; ogni cartella orizzonte (`h1`, `h5`, `h22`) contiene i risultati di Grid Search (`*_gridsearch_*.csv`) e Bayesian Optimization (`*_bayesoptimization_*.csv`), insieme ai grafici di validazione incrociata e previsto-vs-reale.

---

## Dati

- **VIX e regressori di mercato**: S&P 500 (rendimento log), WTI, OVX, DXY, spread OVX-VIX, medie mobili e lag del VIX, RSI a 14 giorni — scaricati da Yahoo Finance tramite `yfinance` (`download_data.py`).
- **Variabili macro da FRED**: T-Bill 3 mesi, Treasury yield 10 anni (→ yield curve), credit spread, richieste di disoccupazione (`initial_claims`) — CSV in `0_Dataset/Data/Raw/`.
- **Polymarket**: probabilità implicite di mercati predittivi su tre temi macro rilevanti per la volatilità — decisioni sui tassi FOMC, probabilità di recessione USA, inflazione CPI YoY — scaricate tramite le API pubbliche di Polymarket (`download_data_polymarket.py`), pulite in due passaggi (`clean_polymarket.py`, `clean_polymarket_2.py`) e trasformate in feature (`feature_engineering_polymarket.py`).
- Il dataset finale con le sole feature tradizionali è `0_Dataset/Data/Clean/dataset.csv`; quello arricchito con le feature di Polymarket è `0_Dataset/Polymarket_data/dataset_w_polymarket.csv`.

---

## Metodologia

1. **Raccolta e pulizia dei dati** (`0_Dataset/`).
2. **Modellazione** su tre orizzonti (1, 5, 22 giorni), per ciascuna delle quattro famiglie di modelli, con ottimizzazione degli iperparametri sia via **Grid Search** sia via **Bayesian Optimization** (tramite `scikit-optimize` / `optuna`).
3. Per i modelli econometrici, di Machine Learning e Deep Learning, ogni run viene ripetuta **con e senza le feature Polymarket** per isolarne il contributo informativo.
4. Per Random Forest e XGBoost è disponibile anche una variante `v2` che utilizza tutti i lag del VIX fino a 22 giorni, invece dei soli lag 1/5/22.
5. **Valutazione comparativa finale** (`6_Final_Results/`): metriche di errore puntuale, Directional Accuracy, analisi per regime di volatilità, test di Diebold-Mariano tra le coppie con/senza Polymarket, e Model Confidence Set tra tutti i modelli.

---

## Risultati

I risultati dettagliati - per modello, orizzonte (1/5/22 giorni) e con/senza feature Polymarket - sono riportati come grafici e CSV nelle cartelle `5_Forecasts_&_Error_Metrics/` e `6_Final_Results/`, e discussi nel testo della tesi.

