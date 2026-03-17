import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

# --------------------------
# TITLE
# --------------------------
st.title("🏠 House Price Predictor")
st.write("AI-powered system to estimate property value")

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("housing_price_model.pkl")

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sq ft)", 500, 20000, 5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)
parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)

main_road = st.sidebar.selectbox("Main Road", [0,1])
guest_room = st.sidebar.selectbox("Guest Room", [0,1])
basement = st.sidebar.selectbox("Basement", [0,1])
hot_water_heating = st.sidebar.selectbox("Hot Water Heating", [0,1])
air_conditioning = st.sidebar.selectbox("Air Conditioning", [0,1])
preferred_area = st.sidebar.selectbox("Preferred Area", [0,1])

furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# --------------------------
# PREP DATA
# --------------------------
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
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --------------------------
# PREDICT BUTTON
# --------------------------
if st.sidebar.button("Predict Price"):

    prediction = model.predict(input_df)[0]

    # NORMALISE VALUE FOR GAUGE (0–100)
    score = min(int(prediction / 100000), 100)

    # --------------------------
    # TOP METRICS (LIKE YOUR SAMPLE)
    # --------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Estimated Price", f"₹{prediction:,.0f}")

    with col2:
        if prediction > 7000000:
            st.success("Premium Property")
            risk = "Low"
        elif prediction > 4000000:
            st.warning("Mid-range Property")
            risk = "Medium"
        else:
            st.error("Low Value Property")
            risk = "High"

    with col3:
        st.metric("Value Score", f"{score}%")

    st.divider()

    # --------------------------
    # GAUGE CHART (LIKE CHURN DASHBOARD)
    # --------------------------
    st.subheader("Price Indicator")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Price Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # INPUT SUMMARY (RIGHT SIDE STYLE)
    # --------------------------
    st.subheader("Key Input Summary")

    summary = pd.DataFrame({
        "Feature":[
            "Area","Bedrooms","Bathrooms","Stories",
            "Parking","Main Road","Guest Room",
            "Basement","Hot Water Heating",
            "Air Conditioning","Preferred Area"
        ],
        "Value":[
            area,bedrooms,bathrooms,stories,
            parking,main_road,guest_room,
            basement,hot_water_heating,
            air_conditioning,preferred_area
        ]
    })

    st.dataframe(summary, use_container_width=True)