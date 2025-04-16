import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("file.csv")

plt.figure(figsize=(12, 6))
sns.boxplot(x="Crop", y="Annual_Rainfall", data=df, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Box Plot: Rainfall Distribution Across Crops")
plt.xlabel("Crop")
plt.ylabel("Annual Rainfall (mm)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
