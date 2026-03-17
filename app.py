import streamlit as st
import pandas as pd
import joblib

# =========================================
# 🏠 PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠"
)

st.title("🏠 House Price Predictor")

# =========================================
# 📦 LOAD TRAINED MODEL
# =========================================
model = joblib.load("housing_price_model.pkl")

# =========================================
# 🧾 USER INPUT (SIDEBAR)
# =========================================
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sq ft)", 500, 20000, 5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)
parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)

# Binary categorical inputs (must match dataset)
mainroad = st.sidebar.selectbox("Main Road", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
basement = st.sidebar.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])

# Multi-category input
furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# =========================================
# 🛠️ PREPARE INPUT DATA
# =========================================
data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

# =========================================
# 🔄 ENCODE CATEGORICAL VARIABLES
# =========================================

# Convert yes/no to 1/0
binary_map = {"yes": 1, "no": 0}

for col in [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]:
    data[col] = data[col].map(binary_map)

# One-hot encode furnishing status
data = pd.get_dummies(data, columns=["furnishingstatus"])

# Ensure all expected dummy columns exist
for col in [
    "furnishingstatus_furnished",
    "furnishingstatus_semi-furnished",
    "furnishingstatus_unfurnished"
]:
    if col not in data:
        data[col] = 0

# Match model feature order
data = data.reindex(columns=model.feature_names_in_, fill_value=0)

# =========================================
# 🔮 MAKE PREDICTION
# =========================================
if st.sidebar.button("Predict Price"):
    prediction = model.predict(data)[0]

    st.subheader("Predicted House Price")
    st.write(f"₹{prediction:,.0f}")

# =========================================
# 📊 INPUT SUMMARY TABLE
# =========================================
st.subheader("Input Summary")

summary = pd.DataFrame({
    "Feature": [
        "Area", "Bedrooms", "Bathrooms", "Stories",
        "Parking", "Main Road", "Guest Room",
        "Basement", "Hot Water Heating",
        "Air Conditioning", "Preferred Area", "Furnishing"
    ],
    "Value": [
        area, bedrooms, bathrooms, stories,
        parking, mainroad, guestroom,
        basement, hotwaterheating,
        airconditioning, prefarea, furnishingstatus
    ]
})

st.table(summary)