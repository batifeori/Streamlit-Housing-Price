# app.py
import streamlit as st
import pandas as pd
import joblib

# --- App Title ---
st.title("House Price Predictor")

# --- Load the trained model ---
model = joblib.load("housing_price_model.pkl")

# --- User Inputs ---
st.header("Enter House Details")

# Numeric features
area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000)
bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
stories = st.slider("Stories", min_value=1, max_value=4, value=2)
parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=1)

# Binary features (0 = No, 1 = Yes)
main_road = st.selectbox("Main Road", [0, 1])
guest_room = st.selectbox("Guest Room", [0, 1])
basement = st.selectbox("Basement", [0, 1])
hot_water_heating = st.selectbox("Hot Water Heating", [0, 1])
air_conditioning = st.selectbox("Air Conditioning", [0, 1])
preferred_area = st.selectbox("Preferred Area", [0, 1])

# Multi-class categorical feature
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# --- Prepare Input DataFrame ---
data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "main_road": main_road,
    "guest_room": guest_room,
    "basement": basement,
    "hot_water_heating": hot_water_heating,
    "air_conditioning": air_conditioning,
    "parking": parking,
    "preferred_area": preferred_area,
    # Manual one-hot encoding for furnishing status
    "furnishingstatus_semi_furnished": 0,
    "furnishingstatus_unfurnished": 0
}

# Set correct one-hot column
if furnishingstatus == "semi-furnished":
    data["furnishingstatus_semi_furnished"] = 1
elif furnishingstatus == "unfurnished":
    data["furnishingstatus_unfurnished"] = 1
# "furnished" leaves both as 0

# Create DataFrame
input_df = pd.DataFrame([data])

# Reindex to match model columns exactly
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --- Predict ---
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ₹{prediction:,.2f}")