import yfinance as yf
import pandas as pd
import numpy as np


dataset = pd.DataFrame()

# S&P500 log return 
sp500_data = yf.download("^GSPC", start = "2014-01-01", end = "2026-06-30")
sp500_data.columns = sp500_data.columns.droplevel(1)
dataset["Close_S&P500"] = sp500_data["Close"]

dataset["Log_Return_S&P500"] = np.log(dataset["Close_S&P500"] / dataset["Close_S&P500"].shift(1)) # con shift di 1, la riga t diventa il valore della chiusura di t-1
dataset = dataset.dropna()


# WTI log return 
wti_data = yf.download("CL=F", start = "2014-01-01", end = "2026-06-30")
wti_data.columns = wti_data.columns.droplevel(1)
dataset["Close_WTI"] = wti_data["Close"]

dataset["Log_Return_WTI"] = np.log(dataset["Close_WTI"] / dataset["Close_WTI"].shift(1))
dataset = dataset.dropna()


# OVX close  
ovx_data = yf.download("^OVX", start = "2014-01-01", end = "2026-06-30")
ovx_data.columns = ovx_data.columns.droplevel(1)
ovx_data = ovx_data.dropna()
dataset["Close_OVX"] = ovx_data["Close"]


# VIX close
vix_data = yf.download("^VIX", start = "2014-01-01", end = "2026-06-30")
vix_data.columns = vix_data.columns.droplevel(1)
vix_data = vix_data.dropna()
dataset["Close_VIX"] = vix_data["Close"]


# VIX lagged values
dataset["VIX_lag1"] = dataset["Close_VIX"].shift(1)
dataset["VIX_lag5"] = dataset["Close_VIX"].shift(5)
dataset["VIX_lag22"] = dataset["Close_VIX"].shift(22)

dataset = dataset.dropna()


# OVX - VIX spread
dataset["OVX_VIX_spread"] = dataset["Close_OVX"] - dataset["Close_VIX"]


# Medie mobili VIX
dataset["VIX_MA5"] = dataset["Close_VIX"].rolling(5).mean()
dataset["VIX_MA10"] = dataset["Close_VIX"].rolling(10).mean()
dataset["VIX_MA20"] = dataset["Close_VIX"].rolling(20).mean()

dataset = dataset.dropna()


# RSI 14 giorni su S&P500
delta = dataset["Close_S&P500"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14, min_periods=14).mean()
avg_loss = loss.rolling(window=14, min_periods=14).mean()
rs = avg_gain / avg_loss
dataset["RSI_14d"] = 100 - (100 / (1 + rs))

dataset = dataset.dropna()


# DXY log returns
dxy_data = yf.download("DX-Y.NYB", start = "2014-01-01", end = "2026-06-30")
dxy_data.columns = dxy_data.columns.droplevel(1)
dataset["Close_DXY"] = dxy_data["Close"]

dataset["Log_Return_DXY"] = np.log(dataset["Close_DXY"] / dataset["Close_DXY"].shift(1)) 

dataset = dataset.dropna()


# Rendimento (yield) treasury americani 10y e 3m
tbill_3m = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\Dataset\Data\Raw\tbill_3m.csv", parse_dates = ["observation_date"])
tbill_3m.rename(columns={"observation_date" : "Date", "TB3MS" : "Close_Tbill_3m"}, inplace = True)
tbill_3m["Date"] = pd.to_datetime(tbill_3m["Date"])
tbill_3m = tbill_3m.sort_values("Date")
tbill_3m = tbill_3m.set_index("Date")
tbill_3m = tbill_3m.asfreq("D")
tbill_3m["Close_Tbill_3m"] = tbill_3m["Close_Tbill_3m"].ffill()

dataset["Close_Tbill_3m"] = tbill_3m["Close_Tbill_3m"].reindex(dataset.index).ffill()

tyield_10y = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\Dataset\Data\Raw\tyield_10y.csv", parse_dates = ["observation_date"])
tyield_10y.rename(columns={"observation_date" : "Date", "DGS10" : "Close_Tyield_10y"}, inplace = True)
tyield_10y = tyield_10y.set_index("Date")

dataset["Close_Tyield_10y"] = tyield_10y["Close_Tyield_10y"]
dataset = dataset.dropna()


# Yield curve (10Y - 3M)
dataset["Yield_curve"] = dataset["Close_Tyield_10y"] - dataset["Close_Tbill_3m"]


# Credit Spread (Baa Corporate Bond - 10-year Treasury) (Moody's)
credit_spread = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\Dataset\Data\Raw\credit_spread.csv", parse_dates = ["observation_date"])
credit_spread.rename(columns={"observation_date" : "Date", "BAA10Y" : "Credit_Spread"}, inplace = True)
credit_spread["Date"] = pd.to_datetime(credit_spread["Date"])
credit_spread = credit_spread.sort_values("Date")
credit_spread = credit_spread.set_index("Date")
credit_spread = credit_spread.asfreq("D")
credit_spread["Credit_Spread"] = credit_spread["Credit_Spread"].ffill()

dataset["Credit_Spread"] = credit_spread["Credit_Spread"].reindex(dataset.index).ffill()


# Jobless claims settimanali 
initial_claims = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\Dataset\Data\Raw\initial_claims.csv", parse_dates = ["observation_date"])
initial_claims.rename(columns = {"observation_date" : "Date", "ICSA" : "Close_Initial_claims"}, inplace = True)
initial_claims["Date"] = pd.to_datetime(initial_claims["Date"])
initial_claims = initial_claims.sort_values("Date")
initial_claims = initial_claims.set_index("Date")
initial_claims = initial_claims.asfreq("D")
#initial_claims["Close_Initial_claims"] = initial_claims["Close_Initial_claims"].ffill()
initial_claims["Log_difference_initial_claims"] = np.log(initial_claims["Close_Initial_claims"] / initial_claims["Close_Initial_claims"].shift(7)) # shift di 7 perchè i dati si aggiornano solo una volta a settimana
initial_claims["Log_difference_initial_claims"] = initial_claims["Log_difference_initial_claims"].ffill()

dataset["Log_difference_initial_claims"] = initial_claims["Log_difference_initial_claims"]


# Gold
gold_data = yf.download("GC=F", start = "2014-01-01", end = "2026-06-30")
gold_data.columns = gold_data.columns.droplevel(1)
dataset["Close_Gold"] = gold_data["Close"]

dataset["Log_Return_Gold"] = np.log(dataset["Close_Gold"] / dataset["Close_Gold"].shift(1)) 
dataset = dataset.dropna()


# Giorno della settimana
dataset["Weekday"] = dataset.index.dayofweek # 0 = Lunedì; 1 = Martedì; ...; Venerdì = 4


# Salvo dataset creato
dataset = dataset.dropna()
dataset.to_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\2. Machine Learning\Dataset\Data\Clean\dataset.csv", index = True)
