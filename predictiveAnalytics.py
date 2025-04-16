import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("file.csv") 

plt.figure(figsize=(12, 6))
sns.stripplot(x="Crop", y="Fertilizer", data=df, hue="Crop", palette="Set2", jitter=True, size=4, dodge=True)
plt.xticks(rotation=90)
plt.title("Fertilizer Use Across Crops")
plt.xlabel("Crop")
plt.ylabel("Fertilizer (kg/hectare)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend([], [], frameon=False)  
plt.show()
