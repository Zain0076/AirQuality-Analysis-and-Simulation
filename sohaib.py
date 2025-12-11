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
