import pandas as pd


evento = "cpi_september_2025"
cartella = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\02. CPI YoY" # inserire nuova cartella ogni volta che si runna il codice
file_output = f"{cartella}/{evento}.csv"

# Carica il dataset originale
df = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\02. CPI YoY\cpi_september_2025.csv")

# Raggruppa per data e unisci i valori non nulli di ciascuna colonna
df_clean = df.groupby("datetime_utc", as_index=False).first()

# Salva il nuovo file CSV
df_clean.to_csv(file_output, index=False)