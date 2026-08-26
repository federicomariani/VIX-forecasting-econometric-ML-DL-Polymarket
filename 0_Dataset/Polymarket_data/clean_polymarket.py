import pandas as pd
import glob
import functools
import os



# ------------------------------------------
# CREAZIONE DATASET MERGED DI OGNI CARTELLA
# ------------------------------------------
# Prima di unire tutti i dataset, ho rinominato la colonna "price" presente in ogni file, in modo da permettere l'unione
# Per ogni file CSV presente in una cartella, li va ad unire in un unico file CSV finale con metodo "outer" con colonna in comune "datetime_utc"
evento = "cpi_may_2026"
cartella = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\02. CPI YoY\2026_may-inflation-us-annual" # inserire nuova cartella ogni volta che si runna il codice
colonna_chiave = "datetime_utc"
file_output = f"{cartella}/{evento}.csv"

 
file_csv = glob.glob(f"{cartella}/*.csv") # prende tutti i file CSV all'interno della cartella

for f in file_csv:
    print(f)
    print("Esiste:", os.path.exists(f))
    print("Dimensione:", os.path.getsize(f) if os.path.exists(f) else "N/A")
    print()
 
dataframes = [
    pd.read_csv(f).drop(columns=["timestamp_unix"], errors="ignore")
    for f in file_csv
] # trasforma CSV in DataFrame ed elimina una colonna non utile 
 
df_finale = functools.reduce(
    lambda left, right: pd.merge(left, right, on=colonna_chiave, how="outer"),
    dataframes
) # unisce tutti i dataframe presenti nella cartella uno alla volta senza eliminare alcuna riga (outer)
 
df_finale = df_finale.sort_values(colonna_chiave)

df_finale["datetime_utc"] = pd.to_datetime(
    df_finale["datetime_utc"]
).dt.strftime("%Y-%m-%d") 

df_finale.to_csv(file_output, index=False)

print(f"Uniti {len(file_csv)}")


