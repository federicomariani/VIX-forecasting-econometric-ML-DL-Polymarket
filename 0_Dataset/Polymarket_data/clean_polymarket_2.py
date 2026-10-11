import pandas as pd


evento = "cpi_september_2025"
cartella = r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\02. CPI YoY" # insert new folder each time you run this code
file_output = f"{cartella}/{evento}.csv"

# Load the original dataset
df = pd.read_csv(r"C:\Users\fede1\OneDrive - Università degli Studi di Macerata\2_Tesi\3_Codici\3. Capitolo - Metodologia\0. Dataset\Polymarket_data\Clean\02. CPI YoY\cpi_september_2025.csv")

# Group by date and merge the non-null values of each column
df_clean = df.groupby("datetime_utc", as_index=False).first()

# Save the new CSV file
df_clean.to_csv(file_output, index=False)