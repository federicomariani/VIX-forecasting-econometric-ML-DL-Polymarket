import pandas as pd
import glob
import functools
import os



# -----------------------------------------------
# CREATION OF THE MERGED DATASET FOR EACH FOLDER
# -----------------------------------------------
# Before merging all datasets, I renamed the "price" column present in each file so that the merge is possible
# For each CSV file in a folder, it merges them into a single final CSV file using the "outer" method on the common column "datetime_utc"
evento = "cpi_may_2026"
cartella = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\02. CPI YoY\2026_may-inflation-us-annual" # insert a new folder each time you run the code
colonna_chiave = "datetime_utc"
file_output = f"{cartella}/{evento}.csv"

 
file_csv = glob.glob(f"{cartella}/*.csv") # retrieves all CSV files inside the folder

for f in file_csv:
    print(f)
    print("Exists:", os.path.exists(f))
    print("Size:", os.path.getsize(f) if os.path.exists(f) else "N/A")
    print()
 
dataframes = [
    pd.read_csv(f).drop(columns=["timestamp_unix"], errors="ignore")
    for f in file_csv
] # converts CSV files into DataFrames and removes an unnecessary column 
 
df_finale = functools.reduce(
    lambda left, right: pd.merge(left, right, on=colonna_chiave, how="outer"),
    dataframes
) # merges all dataframes in the folder one by one without deleting any rows (outer)
 
df_finale = df_finale.sort_values(colonna_chiave)

df_finale["datetime_utc"] = pd.to_datetime(
    df_finale["datetime_utc"]
).dt.strftime("%Y-%m-%d") 

df_finale.to_csv(file_output, index=False)

print(f"Merged {len(file_csv)} files")


