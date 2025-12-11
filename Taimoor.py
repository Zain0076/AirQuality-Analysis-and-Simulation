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
