# ========================== IMPORT LIBRARIES ==========================
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

# ========================== SUPPRESS WARNINGS ==========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow info and warnings
warnings.filterwarnings("ignore")         # Suppress all other warnings

# ========================== LOAD & PREPROCESS DATA ==========================
df = pd.read_csv("file.csv")
target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k='all')
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# ========================== DEFINE & TRAIN MODELS ==========================
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor()
}

results = {}

print("\n=====  Individual Model Performance =====\n")
for name, model in models.items():
    model.fit(X_train_sel, y_train)
    preds = model.predict(X_test_sel)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name:<15} | MSE: {mse:.4f} | R²: {r2:.4f}")

# ========================== LSTM REGRESSOR ==========================
X_train_lstm = X_train_sel.reshape((X_train_sel.shape[0], 1, X_train_sel.shape[1]))
X_test_lstm = X_test_sel.reshape((X_test_sel.shape[0], 1, X_test_sel.shape[1]))

model_lstm = Sequential([
    Input(shape=(1, X_train_sel.shape[1])),
    LSTM(64, activation='relu'),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model_lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)

y_pred_lstm = model_lstm.predict(X_test_lstm).flatten()
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)
results["LSTM"] = {"MSE": mse_lstm, "R2": r2_lstm}
print(f"LSTM           | MSE: {mse_lstm:.4f} | R²: {r2_lstm:.4f}")

# ========================== VOTING ENSEMBLE ==========================
voting = VotingRegressor(estimators=[
    ('rf', RandomForestRegressor()),
    ('xgb', XGBRegressor()),
    ('ada', AdaBoostRegressor())
])
voting.fit(X_train_sel, y_train)

ensemble_preds = voting.predict(X_test_sel)
ensemble_mse = mean_squared_error(y_test, ensemble_preds)
ensemble_r2 = r2_score(y_test, ensemble_preds)
results["Voting Ensemble"] = {"MSE": ensemble_mse, "R2": ensemble_r2}
print(f"Voting Ensemble | MSE: {ensemble_mse:.4f} | R²: {ensemble_r2:.4f}")

# ========================== RESULTS TABLE ==========================
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='R2', ascending=False)
results_df['Rank'] = range(1, len(results_df) + 1)

print("\n=====  Model Comparison Table (Sorted by R²) =====\n")
print(results_df)

# Save to CSV
results_df.to_csv("model_comparison_regression.csv")

# ========================== COMPARISON R² POINT PLOT ==========================
plt.figure(figsize=(10, 6))
sns.pointplot(x=results_df.index, y='R2', data=results_df, color='green', markers='D', linestyles='--')

plt.title('R² Score Comparison Across Models', fontsize=16)
plt.ylabel('R² Score', fontsize=12)
plt.xlabel('Model', fontsize=12)

# Set dynamic Y-axis scaling based on min/max R²
min_r2 = results_df['R2'].min()
max_r2 = results_df['R2'].max()
plt.ylim(max(0, min_r2 - 0.05), min(1, max_r2 + 0.05))

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
