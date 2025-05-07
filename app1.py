import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import os

# Function to find crop images
def find_image(crop_name, image_folder="crop_images"):
    extensions = [".jpg", ".jpeg", ".png"]
    for ext in extensions:
        image_path = os.path.join(image_folder, crop_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

# Cache model loading
@st.cache_resource
def load_models():
    with open("crop_model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    with open("yield_model.pkl", "rb") as f:
        yield_model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("preprocessor_yield.pkl", "rb") as f:
        preprocessor_yield = pickle.load(f)
    return crop_model, yield_model, preprocessor, preprocessor_yield

crop_model, yield_model, preprocessor, preprocessor_yield = load_models()


crop_code_mapping = {
    'Arecanut': 0, 'Arhar_tur': 1, 'Ash Gourd': 2, 'Bajra': 3, 'Banana': 4, 'Barley': 5, 'Bean': 6,
    'Beans & Mutter(Vegetable)': 7, 'Beet Root': 8, 'Ber': 9, 'Bhindi': 10, 'Bitter Gourd': 11,
    'Black pepper': 12, 'Blackgram': 13, 'Bottle Gourd': 14, 'Brinjal': 15, 'Cabbage': 16,
    'Cardamom': 17, 'Carrot': 18, 'Cashewnut': 19, 'Castor seed': 20, 'Cauliflower': 21,
    'Citrus Fruit': 22, 'Coconut': 23, 'Colocosia': 24, 'Coriander': 25, 'Cotton(lint)': 26,
    'Cowpea(Lobia)': 27, 'Cucumber': 28, 'Drum Stick': 29, 'Dry chillies': 30, 'Dry ginger': 31,
    'Garlic': 32, 'Ginger': 33, 'Gram': 34, 'Grapes': 35, 'Groundnut': 36, 'Guar seed': 37,
    'Horse-gram': 38, 'Jack Fruit': 39, 'Jobster': 40, 'Jowar': 41, 'Jute': 42, 'Kapas': 43,
    'Khesari': 44, 'Korra': 45, 'Lab-Lab': 46, 'Lemon': 47, 'Lentil': 48, 'Linseed': 49,
    'Maize': 50, 'Mango': 51, 'Masoor': 52, 'Mesta': 53, 'Moong(Green Gram)': 54, 'Moth': 55,
    'Niger seed': 56, 'Onion': 57, 'Orange': 58, 'Paddy': 59, 'Papaya': 60, 'Peas & beans (Pulses)': 61,
    'Pineapple': 62, 'Pome Fruit': 63, 'Pome Granet': 64, 'Potato': 65, 'Pump Kin': 66,
    'Ragi': 67, 'Rajmash Kholar': 68, 'Rapeseed &Mustard': 69, 'Redish': 70,
    'Ribed Guard': 71, 'Rice': 72, 'Ricebean (nagadal)': 73, 'Rubber': 74, 'Safflower': 75,
    'Samai': 76, 'Sannhamp': 77, 'Sapota': 78, 'Sesamum': 79, 'Small millets': 80,
    'Snak Guard': 81, 'Soyabean': 82, 'Sugarcane': 83, 'Sunflower': 84, 'Sweet potato': 85,
    'Tapioca': 86, 'Tea': 87, 'Tobacco': 88, 'Tomato': 89, 'Turmeric': 90,
    'Urad': 91, 'Varagu': 92, 'Water Melon': 93, 'Wheat': 94, 'Perilla': 95, 'Yam': 96
}

state_options = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
                 "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
                 "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
                 "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
                 "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
                 "West Bengal"]

season_options = ["Kharif", "Rabi", "Summer", "Whole Year", "Winter"]


# Set background image
def set_background():
    image_path = "background.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded_img = base64.b64encode(img.read()).decode()
        bg_style = f"""
            <style>
                [data-testid="stAppViewContainer"] {{
                    background: url("data:image/jpg;base64,{encoded_img}") !important;
                    background-size: cover !important;
                    background-position: center !important;
                    background-attachment: fixed !important;
                }}
            </style>
        """
        st.markdown(bg_style, unsafe_allow_html=True)

set_background()

# Page title
st.title("🌾 Crop Prediction and Yield Estimation")

# Input fields (moved from sidebar to center)
st.subheader("Enter Input Parameters")


filtered_crops = {}

# Prediction button
VALID_RANGES = {
    "Nitrogen": (10, 150),
    "Phosphorus": (10, 150),
    "Potassium": (10, 150),
    "Temperature": (10.0, 60.0),
    "Humidity": (10.0, 100.0),
    "pH": (3.5, 9.0),
    "Rainfall": (10.0, 5000.0)
}

# Collect User Inputs
inputs = {}
for label, (min_val, max_val) in VALID_RANGES.items():
    if isinstance(min_val, float):
        step = 0.0001
        fmt = "%.4f"
    else:
        step = 1
        fmt = "%d"
        
    value = st.number_input(
        f"{label}:", 
        min_value=min_val, 
        max_value=max_val, 
        value=min_val,  # ✅ Set default value to the minimum allowed
        step=step, 
        format=fmt
    )

    inputs[label] = value


# Select state and season
state = st.selectbox("Select State", state_options)
season = st.selectbox("Select Season", season_options)

# Confidence threshold
confidence_threshold = st.slider("Confidence Threshold (%)", min_value=0, max_value=100, value=10)



# Predict Button
if st.button("🔍 Predict Crops & Yield"):
    error_messages = []

    # **Validate Inputs**
    for label, (min_val, max_val) in VALID_RANGES.items():
        if not (min_val <= inputs[label] <= max_val):
            error_messages.append(f"❌ <strong>{label}</strong> must be between <strong>{min_val} - {max_val}</strong>.")

    # **If Errors Exist, Show Messages and Stop**
    if error_messages:
        error_message = "<br>".join(error_messages)
        st.markdown(
            f"""
            <div style="
                background-color: #FFDAB9;  /* Light Orange */
                color: #D35400;  /* Dark Orange */
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                border: 3px solid #D35400;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            ">
                🚫 **Invalid Inputs!** Please correct the following:
                <br>{error_message}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()  # **🚫 Stop Execution Immediately**

    # **If Inputs are Valid, Proceed with Prediction**
    df_test = pd.DataFrame([[inputs["Nitrogen"], inputs["Phosphorus"], inputs["Potassium"],
                             inputs["Temperature"], inputs["Humidity"], inputs["pH"], 
                             inputs["Rainfall"], state, season]],
                           columns=['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall', 'State_Name', 'Season'])

    # **Preprocess & Predict Crops**
    X_crop = preprocessor.transform(df_test)
    crop_probs = crop_model.predict_proba(X_crop)[0]
    crop_predictions = {crop: prob * 100 for crop, prob in zip(crop_code_mapping.keys(), crop_probs)}
    crop_predictions = {k: v for k, v in sorted(crop_predictions.items(), key=lambda item: item[1], reverse=True)}

    # **Filter Crops Based on Confidence Threshold (user-defined)**
    
    filtered_crops = {crop: confidence for crop, confidence in crop_predictions.items() if confidence >= confidence_threshold}

    # **If No Crops Exceed Confidence Threshold, Stop Execution**
    if not filtered_crops:
        st.warning(f"🚫 No crops exceed the {confidence_threshold}% confidence threshold.")
        st.stop()

    # **Yield Prediction & Results Table**
    df_yield = pd.DataFrame(columns=["#", "Image", "Crop", "Confidence (%)", "Predicted Yield (kg/ha)"])

    for idx, (crop, confidence) in enumerate(filtered_crops.items(), start=1):
        image_path = find_image(crop)  # Find image for each crop

        df_test["Crop_Code"] = crop_code_mapping.get(crop, -1)  
        X_yield = preprocessor_yield.transform(df_test)
        base_yield = yield_model.predict(X_yield)[0]
        predicted_yield = base_yield * (confidence / 100)

        df_yield.loc[len(df_yield)] = [idx, image_path if image_path else "No Image", crop, f"{confidence:.2f}%", f"{predicted_yield:.2f} kg/ha"]

    # **Display Results**
    st.subheader("🌱 **Predicted Crops**")

    for i in range(len(df_yield)):  
        confidence = float(df_yield.iloc[i]['Confidence (%)'].replace('%', ''))

        # **Determine Background Color Based on Confidence Level**
        bg_color = "#4CAF50" if confidence >= 70 else "#FFC107" if confidence >= 40 else "#FF5733"

        with st.container():
            st.markdown("---")  # **Separator**

            col1, col2 = st.columns([1, 3])

            # **Image Section**
            with col1:
                image_path = df_yield.iloc[i]['Image']
                if image_path and os.path.exists(image_path):
                    st.image(image_path, width=150)
                else:
                    st.warning("🚫 No Image Available")

            # **Crop Details Section**
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 15px; border-radius: 15px; color: white;">
                        <h3 style="margin-bottom: 5px;">{df_yield.iloc[i]['Crop']}</h3>
                        <p><strong>Confidence: {df_yield.iloc[i]['Confidence (%)']}</strong></p>
                        <p><strong>Yield: {df_yield.iloc[i]['Predicted Yield (kg/ha)']}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
