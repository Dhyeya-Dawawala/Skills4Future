import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Air Pollution Analysis",
    page_icon="💨",
    layout="wide"
)

# --- MODEL LOADING ---
try:
    model = joblib.load('aqi_category_classifier.joblib')
except FileNotFoundError:
    st.error("Model file 'aqi_category_classifier.joblib' not found. Please make sure it's in the correct directory.")
    st.stop()

# --- SIDEBAR FOR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Time-Series Detection", "Interactive AQI Classifier"])

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("💨 Comprehensive Air Pollution Analysis Project")
    st.markdown("---")
    st.subheader("Welcome to my Machine Learning Internship Project!")
    st.write(
        "This interactive web app showcases a multi-faceted analysis of air pollution. "
        "The project is divided into two main parts:"
    )
    st.markdown(
        """
        1.  **Time-Series Detection:** Analyzing historical data from a single city to detect trends and periods of high pollution.
        2.  **Interactive AQI Classifier:** Using a global dataset to build a machine learning model that predicts the AQI category based on pollutant levels.
        """
    )

# --- PAGE 2: TIME-SERIES DETECTION ---
elif page == "Time-Series Detection":
    st.title("Time-Series Analysis: Detecting High Pollution Periods")
    st.markdown("---")
    st.write(
        "This model uses a **Moving Average Crossover** strategy to identify when pollution is trending higher than its historical norm."
    )
    try:
        image = Image.open('moving_average_plot.png')
        st.image(image, caption='Moving Average Crossover Detection', use_container_width=True)
    except FileNotFoundError:
        st.warning("Plot image 'moving_average_plot.png' not found. Please save the plot from your notebook into the project folder.")

# --- PAGE 3: INTERACTIVE AQI CLASSIFIER ---
elif page == "Interactive AQI Classifier":
    st.title("Interactive Air Quality Index (AQI) Classifier")
    st.markdown("---")
    st.sidebar.header("Input Pollutant Values")

    co_val = st.sidebar.slider("CO AQI Value", 0, 100, 5)
    ozone_val = st.sidebar.slider("Ozone AQI Value", 0, 300, 50)
    no2_val = st.sidebar.slider("NO2 AQI Value", 0, 100, 10)
    pm25_val = st.sidebar.slider("PM2.5 AQI Value", 0, 500, 55)

    input_data = pd.DataFrame(
        [[co_val, ozone_val, no2_val, pm25_val]],
        columns=['CO_AQI_Value', 'Ozone_AQI_Value', 'NO2_AQI_Value', 'PM2.5_AQI_Value']
    )

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Model Prediction:")
    if prediction == "Good":
        st.success(f"Predicted AQI Category: **Good**")
    elif prediction == "Moderate":
        st.warning(f"Predicted AQI Category: **Moderate**")
    else:
        st.error(f"Predicted AQI Category: **{prediction}**")
    
    st.write("Prediction Probabilities:")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.dataframe(proba_df)