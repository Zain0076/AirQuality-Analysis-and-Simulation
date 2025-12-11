# uploading csv dataset file from locall drive.
from google.colab import files
uploaded = files.upload()

#loading/mounting csv datset file in colab.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
%matplotlib inline

# Loading our dataset uploaded CSV
df = pd.read_csv("AirQuality.csv", sep=';')
print(df.shape)
df.head()
# 2) ensure Datetime exists (try common names)
if "Datetime" in df.columns:
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
else:
    # try variations
    for cand in ["Date", "date", "Time", "time", "Date;Time", "date;time"]:
        if cand in df.columns:
            df["Datetime"] = pd.to_datetime(df[cand], errors="coerce")
            break

    # if still missing, try building from Date + Time
    if "Datetime" not in df.columns and ("Date" in df.columns and "Time" in df.columns):
        df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")

# If still missing, use the first column as datetime attempt
if "Datetime" not in df.columns:
    df["Datetime"] = pd.to_datetime(df.iloc[:,0], errors="coerce")

# 3) show which gas columns we have
expected = ["CO(GT)","NO2(GT)","NOx(GT)","C6H6(GT)"]
found = [c for c in expected if c in df.columns]
print("Gas columns found:", found)

# 4) make sure gas columns are numeric and -200 were handled earlier, but re-check
df.replace(-200, pd.NA, inplace=True)
for c in found:
    df[c] = pd.to_numeric(df[c], errors="coerce")
#Gas Pollutants Analysis

# Average Gas Concentrations
avg_gases = df[['CO(GT)', 'NOx(GT)', 'NO2(GT)']].mean()
avg_gases.plot(kind='bar', color=['blue','green','red'], figsize=(8,5))
plt.title("Average Gas Pollutant Concentration")
plt.ylabel("Concentration")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['CO(GT)', 'NOx(GT)', 'NO2(GT)']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Gas Pollutants")
plt.show()

# Gas Trends Over Time
for gas in ['CO(GT)', 'NOx(GT)', 'NO2(GT)']:
    plt.figure(figsize=(12,6))
    plt.plot(df['Datetime'], df[gas], marker='o')
    plt.title(f"{gas} Trend Over Time")
    plt.xlabel("Datetime")
    plt.ylabel(f"{gas} Concentration")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
# 12) Save personal output and provide download link
out_name = "taimoor_gas_aqi_alerts.csv"
df.to_csv(out_name, index=False)

from google.colab import files
files.download(out_name)

print(f"\nSaved and started download: {out_name}")
