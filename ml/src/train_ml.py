import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import xgboost as xgb
import joblib 

# 1. Load Data
try:
    df = pd.read_csv('solar_panel_combined_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'solar_panel_combined_dataset.csv' not found.")
    print("Make sure you are in the 'ml/src' folder and the file exists.")
    exit()

# 2. Clean Data & Handle Target Variable
# Create 'Label' (Target for Suitability) - Handle potential casing issues
if 'Label (Yes/No)' in df.columns:
    df['Label'] = df['Label (Yes/No)'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
else:
    # If the column name is different, try to find it or default to 0
    print("Warning: 'Label (Yes/No)' column not found. Checking for similar names...")
    possible_cols = [c for c in df.columns if 'label' in c.lower()]
    if possible_cols:
        print(f"Using '{possible_cols[0]}' as label.")
        df['Label'] = df[possible_cols[0]].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    else:
        print("Error: No Label column found.")
        exit()

# Drop ID and original text label columns if they exist
cols_to_drop = ['panel_id', 'Label (Yes/No)']
df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)

# 3. Handle Categorical Features (One-Hot Encoding)
# This converts any remaining text columns into numbers
df = pd.get_dummies(df, drop_first=True)

# Ensure no missing values (fill with 0 just in case)
df = df.fillna(0)

print(f"Processed dataframe shape: {df.shape}")

# 4. Define Features (X) and Targets (y)
# We need to make sure 'efficiency' and 'Label' are not in X
drop_targets = ['efficiency', 'Label']
X = df.drop([c for c in drop_targets if c in df.columns], axis=1)

y_eff = df['efficiency']
y_site = df['Label']

# 5. Train Efficiency Model (Regressor)
print("\nTraining Efficiency Model...")
X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(X, y_eff, test_size=0.2, random_state=42)

model_eff = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42, objective='reg:squarederror')
model_eff.fit(X_train_eff, y_train_eff)

y_pred_eff = model_eff.predict(X_test_eff)
r2_eff = r2_score(y_test_eff, y_pred_eff)
mae_eff = mean_absolute_error(y_test_eff, y_pred_eff)
print(f"Efficiency Model - R²: {r2_eff:.4f} (Target >0.94), MAE: {mae_eff:.4f}")

# 6. Train Suitability Model (Classifier)
print("\nTraining Suitability Model...")
X_train_site, X_test_site, y_train_site, y_test_site = train_test_split(X, y_site, test_size=0.2, random_state=42)

model_site = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model_site.fit(X_train_site, y_train_site)

y_pred_site = model_site.predict(X_test_site)
acc_site = accuracy_score(y_test_site, y_pred_site)
print(f"Suitability Model Accuracy: {acc_site:.4f}")

# 7. Save Models
joblib.dump(model_eff, "efficiency_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

joblib.dump(model_site, "suitability_model.pkl")
joblib.dump(X.columns.tolist(), "site_feature_columns.pkl")

print("="*100)
print("SUCCESS: All .pkl models generated!")
print("="*100)