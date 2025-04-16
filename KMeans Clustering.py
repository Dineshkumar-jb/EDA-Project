import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("crop_recommendation.csv") 

features = ["Annual_Rainfall", "Fertilizer", "Pesticide", "Yield", "Area", "Production"]  
df_features = df[features]


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)


wcss = []  
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--", color="b")
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("WCSS (Within-cluster sum of squares)", fontsize=12)
plt.title("Elbow Method for Optimal K Selection", fontsize=14)
plt.grid(alpha=0.3)
plt.show()


