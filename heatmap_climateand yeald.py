import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("crop_recommendation.csv")
climate_features = ["Annual_Rainfall", "Fertilizer", "Pesticide", "Yield", "Area", "Production"]
df_climate = df[climate_features]
correlation_matrix = df_climate.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 10})

plt.title("Correlation Heatmap of Climatic Variables and Crop Yield", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
