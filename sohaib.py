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
