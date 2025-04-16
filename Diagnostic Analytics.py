import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("crop_recommendation.csv")
correlation_matrix = df[['Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Climatic Factors vs. Yield")
plt.show()
