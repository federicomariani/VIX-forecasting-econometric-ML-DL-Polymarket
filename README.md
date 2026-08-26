# Previsione del VIX tra modelli econometrici, Machine Learning e Deep Learning: un confronto multi-orizzonte con l’integrazione di dati da Polymarket
Tesi di Laurea Triennale in Economia: Banche, Aziende e Mercati
**Università degli Studi di Macerata** - Dipartimento di Economia e Diritto
Anno Accademico 2025-2026

- **Candidato:** Federico Mariani
- **Relatore:** Prof. Luca Romeo
- **Correlatore:** Prof. Andrea Bucci

## Abstract

Il lavoro confronta modelli di previsione del CBOE Volatility Index (VIX) appartenenti
a quattro famiglie - Random Walk, econometrici (ARIMA, HAR-Type), Machine Learning
(Random Forest, XGBoost) e Deep Learning (LSTM, Temporal Fusion Transformer) - su tre
orizzonti temporali (1, 5 e 22 giorni), utilizzando lo stesso arco temporale per tutti
i modelli. Un contributo originale della tesi è l'integrazione di feature derivanti da
mercati predittivi (Polymarket) nei modelli di Machine Learning e Deep Learning, per
valutare se le probabilità implicite in questi mercati aggiungano informazione utile
alla previsione.

I modelli sono confrontati tramite metriche di accuratezza puntuale (MSE, QLIKE),
Directional Accuracy, test di Diebold-Mariano e Model Confidence Set, anche su diversi
regimi di volatilità (Low / Medium / High).

## Struttura della repository

```
├── 0_dataset/                      # Raccolta e pulizia dei dati
│   ├── data/
│   │   ├── raw/                    # Dati grezzi scaricati (VIX e regressori di mercato)
│   │   ├── clean/                  # Dataset puliti pronti per la modellazione
│   │   └── download_data.py
│   ├── polymarket_data/
│   │   ├── raw/
│   │   ├── clean/
│   │   ├── download_data_polymarket.py
│   │   ├── clean_polymarket.py
│   │   ├── clean_polymarket_2.py
│   │   └── feature_engineering_polymarket.py
│   └── dataset_w_polymarket.csv
│
├── 1_random_walk/                  # Modello benchmark
│   └── random_walk.py
│
├── 2_econometrics/
│   ├── 1_arima/
│   │   └── arima.py
│   └── 2_har_type/
│       └── har_type.py
│
├── 3_machine_learning/
│   ├── 1_random_forest/
│   │   └── random_forest.py
│   └── 2_xgboost/
│       └── xgboost_model.py
│
├── 4_deep_learning/
│   ├── 1_lstm/
│   │   └── lstm.py
│   └── 2_tft/
│       └── tft.py
│
├── 5_model_confidence_set/         # Valutazione comparativa finale
│   ├── mcs.py
│   ├── diebold_mariano.py
│   ├── metriche.py
│   └── regimi_vol.py
│
├── tesi.pdf
├── requirements.txt
└── README.md
```

Ogni cartella dei modelli (2-4) contiene una sottocartella `results/` con le metriche calcolate, 
ed è replicata in una variante `with_polymarket/` per i modelli
addestrati anche con le feature di Polymarket.

## Dati

- **VIX e regressori di mercato** (S&P500, WTI, tassi FED, yield curve, DXY, EPU):
  scaricati da Yahoo Finance tramite `yfinance` e altre fonti pubbliche (`download_data.py`).
- **Polymarket**: dati di prezzo storici dei mercati predittivi rilevanti, scaricati
  tramite le API pubbliche di Polymarket (`download_data_polymarket.py`), puliti e
  trasformati in feature (`feature_engineering_polymarket.py`).

## Librerie principali utilizzate

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `statsmodels`, `boruta`, `shap`,
`torch`, `pytorch-forecasting`, `lightning`, `skorch`, `optuna`, `scikit-optimize`,
`dieboldmariano`, `model-confidence-set`, `matplotlib`, `yfinance`, `requests`.

Vedi [`requirements.txt`](./requirements.txt) per l'elenco completo.
