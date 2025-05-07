# 🌾 Crop Yield Prediction

A machine learning project to predict the most suitable crop and its expected yield based on soil nutrients, weather conditions, and agronomic factors. The system is designed to help farmers and agricultural experts make data-driven decisions.

---

## 📌 Features

- Uses a **Random Forest** model for both classification (crop) and regression (yield)
- Accurate and efficient prediction pipeline
- Input features include agronomic, environmental, and location-based data
- Supports `.pkl` model export for deployment
- Deployed via an interactive **Streamlit** web app

---

## 📂 Dataset

Dataset used: `merged_crop_dataset_filled_complete.csv`

### Key Columns

- **Input Features**:
  - `N`, `P`, `K` – Soil macronutrients
  - `temperature`, `humidity`, `rainfall`, `pH`
  - `season`, `state`, `soil_type`, `fertilizer_use`, `irrigation`, `pest_history`

- **Targets**:
  - `crop` – Most suitable crop (classification)
  - `yield` – Expected yield in tons/hectare (regression)

### Preprocessing

- Missing values filled with domain-reasonable estimates
- Encoded categorical variables appropriately
- Removed zero or inconsistent `yield` values
- Ensured all `season` values are among:  
  `spring`, `summer`, `monsoon`, `autumn`, `winter`, `prevernal`

---

## 🧠 Model Training

- **Model**: `RandomForestClassifier` (for predicting crop)
- **Model**: `RandomForestRegressor` (for predicting yield)
- **Evaluation Metrics**:
  - Classification: Accuracy
  - Regression: R², Mean Squared Error (MSE)
- Final trained model saved as `best_model.pkl` using `pickle`

---

## 🚀 Streamlit App

- **User Inputs**:
  - `N`, `P`, `K`, `pH`, `temperature`, `humidity`, `rainfall`
  - `season`, `state`, `soil_type`, `fertilizer_use`, `irrigation`, `pest_history`

- **Outputs**:
  - Most suitable crop
  - Expected yield

### Run the app:

```bash
streamlit run app.py

