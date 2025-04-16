import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("file.csv") 

def classify_climate(rainfall):
    if rainfall < 500:
        return "Arid"
    elif 500 <= rainfall < 1500:
        return "Temperate"
    else:
        return "Tropical"


df["Climate_Zone"] = df["Annual_Rainfall"].apply(classify_climate)

crop_distribution = df.groupby(["Climate_Zone", "Crop"]).size().unstack()

plt.figure(figsize=(12, 6))
crop_distribution.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12, 6))

plt.title("Crop Distribution Across Climate Zones", fontsize=14)
plt.xlabel("Climate Zone", fontsize=12)
plt.ylabel("Number of Occurrences", fontsize=12)
plt.legend(title="Crop", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
