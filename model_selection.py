import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# ✅ Step 1: Load and Clean the Dataset
df = pd.read_csv("crop_yield_balanced_with_ph.csv")

# ✅ Step 2: Remove Bias-Causing Features
drop_features = ["Rainfall_Temperature", "Weather_Index", "Humidity_Temperature"]
df.drop(columns=drop_features, inplace=True, errors="ignore")

# ✅ Step 3: Define Features and Target Variables
numerical_features = ["N", "P", "K", "rainfall", "humidity", "temperature", "pH"]
categorical_features = ["State_Name", "Season"]
target_crop = "Crop"
target_yield = "Crop Yield (kg per hectare)"

# ✅ Encode Crop Type and Add It to Yield Features
df["Crop_Code"] = df["Crop"].astype("category").cat.codes  # Convert Crop to Numeric

# ✅ Step 4: Preprocessing (Scaling + One-Hot Encoding) **without Crop_Code**
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),  # Exclude Crop_Code
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Apply transformations
X_crop = preprocessor.fit_transform(df.drop(columns=[target_crop, target_yield, "Crop_Code"]))  # Exclude Crop_Code for crop model
y_crop = df[target_crop]

# ✅ Save Preprocessor for Use in Streamlit App
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

# ✅ Step 5: Train-Test Split
X_train, X_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# ✅ Step 6: Train Crop Prediction Model
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_crop_train)

# ✅ Step 7: Train Yield Prediction Model (Now Uses Crop Type)
df["Crop_Code"] = df["Crop"].astype("category").cat.codes  # Ensure Crop_Code is set
numerical_features.append("Crop_Code")  # Add Crop_Code only for yield model

preprocessor_yield = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_yield = preprocessor_yield.fit_transform(df.drop(columns=[target_crop, target_yield]))  # Now include Crop_Code
y_yield = df[target_yield]

X_train_yield, X_test_yield, y_yield_train, y_yield_test = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)

yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model.fit(X_train_yield, y_yield_train)

# ✅ Save the Second Preprocessor (for Yield)
with open("preprocessor_yield.pkl", "wb") as f:
    pickle.dump(preprocessor_yield, f)

# ✅ Save Models
with open("crop_model.pkl", "wb") as f:
    pickle.dump(crop_model, f)

with open("yield_model.pkl", "wb") as f:
    pickle.dump(yield_model, f)

print("✅ Models trained and saved successfully!")
