import pandas as pd
df = pd.read_csv("file.csv")  

def calculate_statistics(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mean = sum(sorted_data) / n

    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]

    min_value = sorted_data[0]
    max_value = sorted_data[-1]
  
    variance = sum((x - mean) ** 2 for x in sorted_data) / n
    std_dev = variance ** 0.5
    
    return mean, median, min_value, max_value, std_dev


numerical_columns = ['Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield', 'Area', 'Production']
stats_results = {}

for col in numerical_columns:
    if col in df.columns:
        stats_results[col] = calculate_statistics(df[col].dropna().tolist())

stats_df = pd.DataFrame.from_dict(stats_results, orient='index', columns=['Mean', 'Median', 'Min', 'Max', 'Std Dev'])
print(stats_df)
