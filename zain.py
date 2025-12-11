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
# Part 3: Visualization and top readings
df['NOx(GT)_Simulated'] = simulated_NOx

plt.figure(figsize=(12,6))
plt.plot(df.index, df['NOx(GT)_Simulated'], color='green', linewidth=2, label='Simulated NOx')
plt.scatter(df.index, df['NOx(GT)_Simulated'], color='darkgreen', s=10)
plt.title("NOx Trend - Simulated Curve")
plt.xlabel("Measurement Index")
plt.ylabel("NOx Concentration (µg/m³)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

top_NOx = df['NOx(GT)_Simulated'].sort_values(ascending=False).head(10)
print("Top NOx readings:\n", top_NOx)
# Part 4: Export and download CSV
out_name = "taimoor_gas_aqi_alerts.csv"
df.to_csv(out_name, index=False)

from google.colab import files
files.download(out_name)
print(f"\nSaved and started download: {out_name}")
