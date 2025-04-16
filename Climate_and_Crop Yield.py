import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("crop_recommendation.csv") 

features = ["Annual_Rainfall", "Fertilizer", "Pesticide", "Yield", "Area", "Production"]  
df_features = df[features]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
df_pca["Crop"] = df["Crop"]  

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="Crop", data=df_pca, palette="viridis", alpha=0.7, edgecolor="k")

plt.title("PCA Visualization of Climate and Crop Yield Data", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Crop", bbox_to_anchor=(1.05, 1), loc="upper left") 
plt.grid(alpha=0.3)

plt.show()
