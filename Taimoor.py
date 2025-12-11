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
