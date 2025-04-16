import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("file.csv")

df["Crop_Year"] = pd.to_datetime(df["Crop_Year"], format="%Y")

yearly_yield = df.groupby("Crop_Year")["Yield"].mean()
decomposition = seasonal_decompose(yearly_yield, model="additive", period=1)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(yearly_yield, label="Original Data", color="blue")
plt.title("Time-Series Decomposition of Crop Yield", fontsize=14)
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label="Trend", color="green")
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label="Seasonality", color="orange")
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label="Residuals (Anomalies)", color="red")
plt.legend()

plt.tight_layout()
plt.show()

