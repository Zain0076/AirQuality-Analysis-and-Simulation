# uploading csv dataset file from local drive
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
%matplotlib inline

# Loading dataset
df = pd.read_csv("AirQuality.csv", sep=';')
print(df.shape)
df.head()

# Ensure Datetime column exists
if "Datetime" in df.columns:
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
else:
    for cand in ["Date", "date", "Time", "time", "Date;Time", "date;time"]:
        if cand in df.columns:
            df["Datetime"] = pd.to_datetime(df[cand], errors="coerce")
            break
    if "Datetime" not in df.columns and ("Date" in df.columns and "Time" in df.columns):
        df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
if "Datetime" not in df.columns:
    df["Datetime"] = pd.to_datetime(df.iloc[:,0], errors="coerce")
# Show which gas columns exist
expected = ["CO(GT)","NO2(GT)","NOx(GT)","C6H6(GT)"]
found = [c for c in expected if c in df.columns]
print("Gas columns found:", found)

# Ensure gas columns are numeric and handle -200
df.replace(-200, pd.NA, inplace=True)
for c in found:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute Gas_AQI using available gas columns
weights = {"CO(GT)":50, "NO2(GT)":20, "NOx(GT)":0.5, "C6H6(GT)":10}
used_cols = [c for c in found if df[c].notna().sum() > 0]
print("Using columns for Gas_AQI:", used_cols)

if not used_cols:
    raise RuntimeError("No usable gas columns with numeric data found. Check file.")

df["Gas_AQI"] = 0.0
for c in used_cols:
    df["Gas_AQI"] += df[c].fillna(0) * weights.get(c, 1.0)
# Classify Gas_AQI into categories
def classify(v):
    if pd.isna(v): return "Unknown"
    v = float(v)
    if v <= 50: return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

df["AQI_Category"] = df["Gas_AQI"].apply(classify)

# Show stats
print("\nGas_AQI summary:")
display(df["Gas_AQI"].describe())
print("\nAQI category counts:")
display(df["AQI_Category"].value_counts(dropna=False))
display(df[["Datetime"] + used_cols + ["Gas_AQI","AQI_Category"]].head(8))

# Plot 1: Gas_AQI over time
plt.figure(figsize=(12,4))
plot_df = df.dropna(subset=["Datetime","Gas_AQI"])
plt.plot(plot_df["Datetime"], plot_df["Gas_AQI"], marker='.', linewidth=0.6)
plt.xlabel("Datetime")
plt.ylabel("Gas_AQI")
plt.title("Gas-Based AQI Over Time")
plt.tight_layout()
plt.show()

# Plot 2: CO distribution
if "CO(GT)" in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df["CO(GT)"].dropna(), bins=40, kde=True)
    plt.title("CO(GT) Distribution")
    plt.xlabel("CO (GT)")
    plt.show()
else:
    print("CO(GT) not present; skipping CO histogram.")

# Plot 3: AQI Category distribution
vc = df["AQI_Category"].value_counts()
vc_plot = vc.drop(labels=["Unknown"], errors='ignore')
if vc_plot.sum() > 0:
    plt.figure(figsize=(8,4))
    vc_plot.plot(kind="bar")
    plt.title("Gas AQI Category Distribution")
    plt.xlabel("AQI Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("No non-Unknown AQI categories to plot.")
