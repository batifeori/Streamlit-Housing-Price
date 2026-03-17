import streamlit as st
import pandas as pd
import joblib

# =========================================
# 🏠 PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Predictor")
st.caption("Smart property valuation using Machine Learning")

# =========================================
# 📦 LOAD MODEL
# =========================================
model = joblib.load("housing_price_model.pkl")

# =========================================
# 🧾 SIDEBAR INPUTS
# =========================================
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sq ft)", 500, 20000, 5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)
parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)

st.sidebar.markdown("### Features")

mainroad = st.sidebar.selectbox("Main Road", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
basement = st.sidebar.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

predict_btn = st.sidebar.button("💰 Predict Price")

# =========================================
# 🛠️ PREPARE DATA (UNCHANGED LOGIC)
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

# Encode yes/no → 1/0
binary_map = {"yes": 1, "no": 0}
for col in [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]:
    data[col] = data[col].map(binary_map)

# One-hot encoding
data = pd.get_dummies(data, columns=["furnishingstatus"])

# Ensure all columns exist
for col in [
    "furnishingstatus_furnished",
    "furnishingstatus_semi-furnished",
    "furnishingstatus_unfurnished"
]:
    if col not in data:
        data[col] = 0

# Match training features
data = data.reindex(columns=model.feature_names_in_, fill_value=0)

# =========================================
# 🔮 PREDICTION DISPLAY
# =========================================
if predict_btn:
    prediction = model.predict(data)[0]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Estimated Price", f"₹{prediction:,.0f}")

    with col2:
        if prediction > 5000000:
            st.success("🏆 High Value Property")
        elif prediction > 3000000:
            st.warning("🏠 Mid Range Property")
        else:
            st.info("💡 Affordable Property")

# =========================================
# 📊 VISUAL INPUT SUMMARY (NO TABLE)
# =========================================
st.divider()
st.subheader("📊 Property Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Area", f"{area} sq ft")
    st.metric("Bedrooms", bedrooms)
    st.metric("Bathrooms", bathrooms)

with col2:
    st.metric("Stories", stories)
    st.metric("Parking", parking)
    st.metric("Furnishing", furnishingstatus)

with col3:
    def yes_no_icon(value):
        return "✅ Yes" if value == "yes" else "❌ No"

    st.write(f"**Main Road:** {yes_no_icon(mainroad)}")
    st.write(f"**Guest Room:** {yes_no_icon(guestroom)}")
    st.write(f"**Basement:** {yes_no_icon(basement)}")
    st.write(f"**Air Conditioning:** {yes_no_icon(airconditioning)}")
    st.write(f"**Hot Water Heating:** {yes_no_icon(hotwaterheating)}")
    st.write(f"**Preferred Area:** {yes_no_icon(prefarea)}")