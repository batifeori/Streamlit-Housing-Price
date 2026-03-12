import streamlit as st
import pandas as pd
import joblib

model = joblib.load("housing_price_model.pkl")

st.title("House Price Predictor")

area = st.number_input("Area")
bedrooms = st.slider("Bedrooms",1,10,3)
bathrooms = st.slider("Bathrooms",1,5,2)
stories = st.slider("Stories",1,4,2)

mainroad = st.selectbox("Main Road",[0,1])
guestroom = st.selectbox("Guest Room",[0,1])
basement = st.selectbox("Basement",[0,1])
hotwaterheating = st.selectbox("Hot Water Heating",[0,1])
airconditioning = st.selectbox("Air Conditioning",[0,1])

parking = st.slider("Parking",0,3,1)
prefarea = st.selectbox("Preferred Area",[0,1])

furnishingstatus_semi = st.selectbox("Semi Furnished",[0,1])
furnishingstatus_unfurnished = st.selectbox("Unfurnished",[0,1])

input_df = pd.DataFrame([[area,bedrooms,bathrooms,stories,
mainroad,guestroom,basement,hotwaterheating,airconditioning,
parking,prefarea,furnishingstatus_semi,furnishingstatus_unfurnished]],
columns=['area','bedrooms','bathrooms','stories','mainroad',
'guestroom','basement','hotwaterheating','airconditioning',
'parking','prefarea','furnishingstatus_semi-furnished',
'furnishingstatus_unfurnished'])

if st.button("Predict Price"):

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Price: {prediction:,.2f}")