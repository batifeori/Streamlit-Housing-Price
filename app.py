# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 House Price Predictor")
st.write("Enter house details to estimate the property price.")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("housing_price_model.pkl")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter House Details")

area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000)

bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
stories = st.slider("Stories", 1, 4, 2)
parking = st.slider("Parking Spaces", 0, 3, 1)

# Binary features
main_road = st.selectbox("Main Road", [0, 1])
guest_room = st.selectbox("Guest Room", [0, 1])
basement = st.selectbox("Basement", [0, 1])
hot_water_heating = st.selectbox("Hot Water Heating", [0, 1])
air_conditioning = st.selectbox("Air Conditioning", [0, 1])
preferred_area = st.selectbox("Preferred Area", [0, 1])

# Furnishing
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# -----------------------------
# Prepare Input Data
# -----------------------------
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
    "furnishingstatus_semi_furnished": 0,
    "furnishingstatus_unfurnished": 0
}

if furnishingstatus == "semi-furnished":
    data["furnishingstatus_semi_furnished"] = 1
elif furnishingstatus == "unfurnished":
    data["furnishingstatus_unfurnished"] = 1

input_df = pd.DataFrame([data])

# Match model column order
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted House Price: ₹{prediction:,.0f}")

    # Price interpretation
    if prediction < 3000000:
        st.info("💡 This property is in the lower price range.")
    elif prediction < 7000000:
        st.success("💡 This property is in the mid-range price category.")
    else:
        st.warning("💡 This property is considered high value.")

# -----------------------------
# Model Performance Section
# -----------------------------
st.divider()
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

# Example values (replace with real ones if you computed them)
col1.metric("R² Score", "0.82")
col2.metric("Average Error", "₹540K")

# -----------------------------
# Feature Importance
# -----------------------------
if hasattr(model, "feature_importances_"):

    st.subheader("📈 Feature Importance")

    importance = model.feature_importances_
    features = model.feature_names_in_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# Dataset Visualization
# -----------------------------
st.subheader("📉 Dataset Visualization")

try:
    df = pd.read_csv("Housing.csv")

    fig, ax = plt.subplots()

    ax.scatter(df["area"], df["price"])
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price")

    st.pyplot(fig)

except:
    st.info("Upload Housing.csv to display dataset visualization.")