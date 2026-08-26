import pandas as pd
import numpy as np
import os
import functools




# ---------------
# IMPORT DATASET
# ---------------
df_fomc = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\0. FOMC Interest Rates\df_fomc.csv")
df_us_recession = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\01. US Recession\df_us_recession.csv")
df_cpi = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\02. CPI YoY\df_cpi.csv")

for df in [df_fomc, df_us_recession, df_cpi]:
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)


# Pulizia dei dati l'ho già fatta sia manualmente su Excel che su Python, quindi passo direttamente alla fase di Feature Engineering per ogni dataset


# ---------------------------
# FEATURE ENGINEERING - FOMC
# ---------------------------
# 1. Expected Rate Change
df_fomc["expected_rate_change_fomc"] = df_fomc["cut_50"]*(-50) + df_fomc["cut_25"]*(-25) + df_fomc["hold"]*(0) + df_fomc["hike_25"]*(25)

# 2. Rate Uncertainty
df_fomc["rate_uncertainty_fomc"] = (df_fomc["cut_50"]*(-50 - df_fomc["expected_rate_change_fomc"]) ** 2) + (df_fomc["cut_25"]*(-25 - df_fomc["expected_rate_change_fomc"]) ** 2) + (df_fomc["hold"]*(0 - df_fomc["expected_rate_change_fomc"]) ** 2) + (df_fomc["hike_25"]*(25 - df_fomc["expected_rate_change_fomc"]) ** 2)
df_fomc["rate_uncertainty_fomc"] = (df_fomc["rate_uncertainty_fomc"]) ** 0.5

# 3. Probability of a Cut/Hike
df_fomc["probability_cut_fomc"] = df_fomc["cut_25"] + df_fomc["cut_50"]
df_fomc["probability_hike_fomc"] = df_fomc["hike_25"]

# 4. Shannon Entropy
df_fomc["entropy_fomc"] = - ((df_fomc["cut_50"] * np.log2(df_fomc["cut_50"])) + (df_fomc["cut_25"] * np.log2(df_fomc["cut_25"])) + (df_fomc["hold"] * np.log2(df_fomc["hold"])) + (df_fomc["hike_25"] * np.log2(df_fomc["hike_25"])))

df_fomc = df_fomc.dropna()

# -----------------------------------
# FEATURE ENGINEERING - US RECESSION
# -----------------------------------
# 1. Shannon Entropy
df_us_recession["entropy_us_recession"] = -(df_us_recession["us_recession"] * np.log2(df_us_recession["us_recession"])+ (1 - df_us_recession["us_recession"]) * np.log2(1 - df_us_recession["us_recession"]))

# 2. Change 1 and 5 Days
df_us_recession["change_1d"] = df_us_recession["us_recession"].diff(1)
df_us_recession["change_5d"] = df_us_recession["us_recession"].diff(5)

df_us_recession = df_us_recession.dropna()


# ------------------------------
# FEATURE ENGINEERING - CPI YoY
# ------------------------------
# 1. Expected CPI
df_cpi["expected_cpi"] = df_cpi["cpi_<=2.7"] * 2.7 + df_cpi["cpi_2.8"] * 2.8 + df_cpi["cpi_2.9"] * 2.9 + df_cpi["cpi_>=3.0"] * 3.0

# 2. CPI Uncertainty
df_cpi["uncertainty_cpi"] = (df_cpi["cpi_<=2.7"]*(2.7 - df_cpi["expected_cpi"]) ** 2) + (df_cpi["cpi_2.8"]*(2.8 - df_cpi["expected_cpi"]) ** 2) + (df_cpi["cpi_2.9"]*(2.9 - df_cpi["expected_cpi"]) ** 2) + (df_cpi["cpi_>=3.0"]*(3.0 - df_cpi["expected_cpi"]) ** 2)
df_cpi["uncertainty_cpi"] = df_cpi["uncertainty_cpi"] ** 0.5

# 3. Shannon Entropy
df_cpi["entropy_cpi"] = - ((df_cpi["cpi_<=2.7"] * np.log2(df_cpi["cpi_<=2.7"])) + (df_cpi["cpi_2.8"] * np.log2(df_cpi["cpi_2.8"])) + (df_cpi["cpi_2.9"] * np.log2(df_cpi["cpi_2.9"])) + (df_cpi["cpi_>=3.0"] * np.log2(df_cpi["cpi_>=3.0"])))

df_cpi = df_cpi.dropna()


# --------------
# MERGE DATASET 
# --------------
evento = "df_polymarket"
output_folder = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean" 
file_output = f"{output_folder}/{evento}.csv"

dataframes = [df_fomc, df_us_recession, df_cpi]

for d in dataframes:
    d.index.name = "Date"

df_polymarket = functools.reduce(
    lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"),
    dataframes
)

df_polymarket = df_polymarket.dropna()
df_polymarket = df_polymarket[~df_polymarket.index.duplicated(keep="first")]

df_polymarket.to_csv(file_output, index = True)


# --------------------------------
# MERGE DATASET PM + TRADIZIONALE
# --------------------------------
evento_2 = "df_finale"
output_folder_2 = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset" 
file_output_2 = f"{output_folder_2}/{evento_2}.csv"

df_tradizionale = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Data\Clean\df_x_polymarket.csv")
df_tradizionale["Date"] = pd.to_datetime(df_tradizionale["Date"], dayfirst=True)
df_tradizionale.set_index("Date", inplace=True)
df_tradizionale.sort_index(inplace=True)

dataframes_2 = [df_tradizionale, df_polymarket]

df_finale = functools.reduce(
    lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"),
    dataframes_2
)

df_finale = df_finale.dropna()
df_finale = df_finale[~df_finale.index.duplicated(keep="first")]

df_finale.to_csv(file_output_2, index=True)
