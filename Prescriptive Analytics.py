import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("file.csv") 
from sklearn.cluster import KMeans

optimal_clusters = 4  
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Annual_Rainfall', 'Fertilizer', 'Yield']])

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual_Rainfall'], y=df['Yield'], hue=df['Cluster'], palette="viridis", s=100)
plt.xlabel("Annual Rainfall")
plt.ylabel("Yield")
plt.title("Crop Clusters Based on Climate & Yield")
plt.legend(title="Cluster")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
