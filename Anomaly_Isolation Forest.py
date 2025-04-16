import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("crop_recommendation.csv") 
features = ["Annual_Rainfall", "Fertilizer", "Pesticide", "Yield", "Area", "Production"]
df_features = df[features]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

iso_forest = IsolationForest(contamination=0.05, random_state=42)  
df["Anomaly"] = iso_forest.fit_predict(df_scaled)

df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Annual_Rainfall"], y=df["Yield"], hue=df["Anomaly"], palette={"Normal": "blue", "Anomaly": "red"}, alpha=0.7, edgecolor="k")

plt.title("Anomaly Detection in Crop Yield Using Isolation Forest", fontsize=14)
plt.xlabel("Annual Rainfall (mm)", fontsize=12)
plt.ylabel("Crop Yield (tons per hectare)", fontsize=12)
plt.legend(title="Data Point Type")
plt.grid(alpha=0.3)

plt.show()
