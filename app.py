# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Setup ---
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor & Dashboard")
st.markdown("Predict house prices and explore trends interactively.")

# --- Load trained model ---
model = joblib.load("housing_price_model.pkl")

# --- Load dataset for visualization ---
df = pd.read_csv("housing.csv")  # Replace with your dataset path

# --- Sidebar: User Inputs ---
st.sidebar.header("Enter House Details")

# Numeric features
area = st.sidebar.number_input("Area (sq ft)", 500, 20000, 5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)
parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)

# Binary features
main_road = st.sidebar.selectbox("Main Road", [0, 1])
guest_room = st.sidebar.selectbox("Guest Room", [0, 1])
basement = st.sidebar.selectbox("Basement", [0, 1])
hot_water_heating = st.sidebar.selectbox("Hot Water Heating", [0, 1])
air_conditioning = st.sidebar.selectbox("Air Conditioning", [0, 1])
preferred_area = st.sidebar.selectbox("Preferred Area", [0, 1])

# Categorical feature
furnishingstatus = st.sidebar.selectbox(
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
    "furnishingstatus_semi_furnished": 0,
    "furnishingstatus_unfurnished": 0
}

if furnishingstatus == "semi-furnished":
    data["furnishingstatus_semi_furnished"] = 1
elif furnishingstatus == "unfurnished":
    data["furnishingstatus_unfurnished"] = 1

input_df = pd.DataFrame([data])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --- Prediction ---
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"Predicted House Price: ₹{prediction:,.2f}")

    # Store prediction history in session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.session_state['history'].append(prediction)

# --- Main Dashboard ---
st.header("📊 Dashboard & Data Visualizations")

tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Prediction History", "Feature Analysis"])

# --- Tab 1: Dataset Overview ---
with tab1:
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(df['price'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# --- Tab 2: Prediction History ---
with tab2:
    st.subheader("Prediction History")
    if 'history' in st.session_state and st.session_state['history']:
        hist_df = pd.DataFrame({
            "Prediction": st.session_state['history'],
            "Index": range(1, len(st.session_state['history'])+1)
        })
        st.line_chart(hist_df.set_index("Index"))
        st.dataframe(hist_df)
    else:
        st.info("No predictions yet. Use the sidebar to predict house prices.")

# --- Tab 3: Feature Analysis ---
with tab3:
    st.subheader("Feature vs Price Analysis")
    numeric_features = ['area','bedrooms','bathrooms','stories','parking']
    selected_feature = st.selectbox("Select feature to visualize", numeric_features)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(data=df, x=selected_feature, y='price', alpha=0.6, ax=ax)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Price")
    ax.set_title(f"{selected_feature} vs Price")
    st.pyplot(fig)

st.markdown("---")
st.caption("Developed with Streamlit, Pandas, Scikit-learn, Matplotlib, and Seaborn")