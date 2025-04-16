import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

file_path = "file.csv" 
df = pd.read_csv(file_path)

df = df.sort_values(by="Crop_Year")

years = df["Crop_Year"].values
rainfall = df["Annual_Rainfall"].values
yield_data = df["Yield"].values


algo = rpt.Pelt(model="l2").fit(rainfall)
change_points = algo.predict(pen=10)  

plt.figure(figsize=(12, 5))
plt.plot(years, rainfall, label="Annual Rainfall", color="blue")
for cp in change_points[:-1]: 
    plt.axvline(years[cp], color="red", linestyle="--", label="Change Point" if cp == change_points[0] else "")

plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.title("Change Point Detection in Rainfall Trends")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(years, yield_data, label="Yield", color="green")
for cp in change_points[:-1]:
    plt.axvline(years[cp], color="red", linestyle="--", label="Change Point" if cp == change_points[0] else "")

plt.xlabel("Year")
plt.ylabel("Crop Yield")
plt.title("Impact of Climate Change on Crop Yield")
plt.legend()
plt.grid()
plt.show()
