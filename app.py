import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Air Pollution Analysis",
    page_icon="💨",
    layout="wide"
)



@st.cache_data
def load_data(file_path):
    """Loads data from a CSV file. Cached to prevent reloading."""
    return pd.read_csv(file_path)

@st.cache_resource
def train_and_get_model():
    """
    This is the key function. It trains the model ONCE and caches it.
    This guarantees no version mismatch between training and prediction.
    """
   
    try:
        df_train = load_data('air pollution dataset.csv')
    except FileNotFoundError:
        st.error("Training data 'air pollution dataset.csv' not found in the project folder. Cannot build the app.")
        return None

    
    df_train.columns = df_train.columns.str.replace(' ', '_')
    
    
    features = ['CO_AQI_Value', 'Ozone_AQI_Value', 'NO2_AQI_Value', 'PM2.5_AQI_Value']
    target = 'AQI_Category'
    
   
    df_train.dropna(subset=features + [target], inplace=True)
    
    X = df_train[features]
    y = df_train[target]
    

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    st.info("Training a new classification model... (This will only happen on the first run)")
    model.fit(X, y)
    st.success("Model training complete!")
    
    return model


try:
    df_global = load_data('air pollution dataset.csv')
    model = train_and_get_model()
except Exception as e:
    st.error(f"An error occurred during data loading or model training: {e}")
    st.stop() 

if model is None:
    st.stop()



st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Global Pollution Insights", "Interactive AQI Classifier"])


if page == "Project Overview":
    st.title("💨 Comprehensive Air Pollution Analysis Project")
    st.markdown("---")
    st.subheader("Welcome to my Machine Learning Internship Project!")
    st.write("This interactive web app showcases an analysis of global air pollution and features an interactive model to predict Air Quality.")
    st.info("Use the navigation panel on the left to explore the different parts of the project.")


elif page == "Global Pollution Insights":
    st.title("Global Insights: Comparing Pollution Across Cities")
    st.markdown("---")
    st.write("This section uses a global dataset to find pollution hotspots and understand which pollutants are most common.")
    
    st.subheader("Top 10 Most Polluted Cities by Average AQI")
    
    # --- Top 10 Cities Plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_10_polluted_cities = df_global.groupby('City')['AQI Value'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_10_polluted_cities.values, y=top_10_polluted_cities.index, ax=ax1, palette='plasma')
    ax1.set_xlabel("Average AQI Value")
    ax1.set_ylabel("City")
    st.pyplot(fig1)

    st.subheader("Global Distribution of AQI Categories")

    # --- AQI Category Distribution Plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    category_order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    sns.countplot(x='AQI Category', data=df_global, order=category_order, ax=ax2, palette='viridis')
    ax2.set_xlabel("AQI Category")
    ax2.set_ylabel("Number of Cities")
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig2)


elif page == "Interactive AQI Classifier":
    st.title("Interactive Air Quality Index (AQI) Classifier")
    st.markdown("---")
    st.sidebar.header("Input Pollutant Values")
    
    co_val = st.sidebar.slider("CO AQI Value", 0, 100, 5, key="co")
    ozone_val = st.sidebar.slider("Ozone AQI Value", 0, 300, 50, key="ozone")
    no2_val = st.sidebar.slider("NO2 AQI Value", 0, 100, 10, key="no2")
    pm25_val = st.sidebar.slider("PM2.5 AQI Value", 0, 500, 55, key="pm25")

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