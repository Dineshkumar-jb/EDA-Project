import pandas as pd

df = pd.read_csv("crop_recommendation.csv")  

print("Dataset Overview:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())
