# Part 1: Upload and basic cleaning
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np

df = pd.read_csv("AirQuality.csv", sep=';', decimal=',')
df = df.iloc[:, :-2]  # drop last empty columns if exist

print(df.head())

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Datetime'])

expected = ["CO(GT)","NO2(GT)","NOx(GT)","C6H6(GT)"]
found = [c for c in expected if c in df.columns]
print("Gas columns found:", found)
# Part 2: Reload dataset and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("AirQuality.csv", sep=';', decimal=',')
df = df.iloc[:, :-2]  # drop empty columns

n = len(df)

time_index = np.arange(n)
simulated_NOx = 50 + 30*np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 5, n)
simulated_NOx[simulated_NOx < 0] = 0
