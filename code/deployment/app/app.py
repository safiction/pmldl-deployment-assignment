"""Stage 3: web app.

Input fields for the model's features, a button to run a prediction, and an
area showing the result returned by the API.

Run from the repository root:  streamlit run code/deployment/app/app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

st.title("Walmart Weekly Sales Predictor")

store = st.number_input("Store", min_value=1, max_value=45, value=1, step=1)
holiday_flag = st.checkbox("Holiday week")
temperature = st.number_input("Temperature (F)", value=60.0)
fuel_price = st.number_input("Fuel price", min_value=0.0, value=3.0)
cpi = st.number_input("CPI", min_value=0.0, value=180.0)
unemployment = st.number_input("Unemployment rate", min_value=0.0, value=8.0)
year = st.number_input("Year", min_value=2000, max_value=2100, value=2012, step=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)
week = st.number_input("Week", min_value=1, max_value=53, value=1, step=1)

if st.button("Predict weekly sales"):
    payload = {
        "store": int(store),
        "holiday_flag": int(holiday_flag),
        "temperature": temperature,
        "fuel_price": fuel_price,
        "cpi": cpi,
        "unemployment": unemployment,
        "year": int(year),
        "month": int(month),
        "week": int(week),
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()["predicted_weekly_sales"]
        st.success(f"Predicted weekly sales: ${prediction:,.2f}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
