import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# === Load Dataset ===
df = pd.read_csv("file.csv")

# === Clean Missing Values ===
numeric_cols = ['Production', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in ['State', 'Season', 'Crop']:
    df[col] = df[col].fillna(df[col].mode()[0])

# === Normalize text columns to avoid input mismatch ===
df['State'] = df['State'].str.strip().str.lower()
df['Season'] = df['Season'].str.strip().str.lower()
df['Crop'] = df['Crop'].str.strip()

# === Compute Yield & Score ===
df['Yield'] = df['Production'] / df['Area']
df['Crop_Score'] = df['Yield'] - (0.5 * df['Fertilizer'] + 0.5 * df['Pesticide'])

# === Get best crop per (State, Season) ===
grouped = df.groupby(['State', 'Season', 'Crop'])['Crop_Score'].mean().reset_index()
best_crops = grouped.loc[grouped.groupby(['State', 'Season'])['Crop_Score'].idxmax()]

# === Prepare for modeling ===
df_model = best_crops[['State', 'Season', 'Crop']]

# === Label Encoding ===
le_state = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df_model['State_enc'] = le_state.fit_transform(df_model['State'])
df_model['Season_enc'] = le_season.fit_transform(df_model['Season'])
df_model['Crop_enc'] = le_crop.fit_transform(df_model['Crop'])

# === Train model ===
X = df_model[['State_enc', 'Season_enc']]
y = df_model['Crop_enc']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === User input ===
print("\n--- Crop Suggestion System ---")
state_input = input("Enter your State: ").strip().lower()
season_input = input("Enter the Season: ").strip().lower()

# === Validate input ===
valid_states = df_model['State'].unique()
valid_seasons = df_model['Season'].unique()

if state_input not in valid_states:
    print(f"\n Invalid State: '{state_input}'\nAvailable States: {sorted(set(valid_states))}")
elif season_input not in valid_seasons:
    print(f"\n Invalid Season: '{season_input}'\nAvailable Seasons: {sorted(set(valid_seasons))}")
else:
    state_encoded = le_state.transform([state_input])[0]
    season_encoded = le_season.transform([season_input])[0]
    input_data = pd.DataFrame([[state_encoded, season_encoded]], columns=['State_enc', 'Season_enc'])

    crop_encoded = model.predict(input_data)[0]
    suggested_crop = le_crop.inverse_transform([crop_encoded])[0]

    print(f"\n Suggested Crop for '{state_input.title()}' in '{season_input.title()}': **{suggested_crop}**")
    print(" Based on high yield, low pesticide and fertilizer usage.")
