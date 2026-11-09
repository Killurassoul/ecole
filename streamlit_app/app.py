import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

st.set_page_config(page_title="Machine Learning — Classification et Régression", page_icon="🤖", layout="centered")


@st.cache_resource
def load_objects():
    classifier = joblib.load(ARTIFACTS_DIR / "best_classifier.joblib")
    regressor = joblib.load(ARTIFACTS_DIR / "best_regressor.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return classifier, regressor, metadata


classifier, regressor, metadata = load_objects()

st.title("🤖 Application de Machine Learning")
st.write("Cette application utilise uniquement le meilleur modèle de classification et le meilleur modèle de régression, sélectionnés après comparaison dans les notebooks.")

tab_classification, tab_regression = st.tabs(["🌸 Classification", "🚕 Régression"])

with tab_classification:
    st.subheader("Prédire l'espèce d'une fleur Iris")
    st.caption(f"Modèle déployé : {metadata['classification']['best_model']}")
    st.write("Saisis les mesures de la fleur en centimètres.")
    with st.form("classification_form"):
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Longueur du sépale", min_value=0.0, value=5.8, step=0.1)
            petal_length = st.number_input("Longueur du pétale", min_value=0.0, value=4.0, step=0.1)
        with col2:
            sepal_width = st.number_input("Largeur du sépale", min_value=0.0, value=3.0, step=0.1)
            petal_width = st.number_input("Largeur du pétale", min_value=0.0, value=1.2, step=0.1)
        classify = st.form_submit_button("Prédire l'espèce")

    if classify:
        input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=metadata["classification"]["features"])
        prediction = classifier.predict(input_df)[0]
        st.success(f"Espèce prédite : **{prediction}**")

with tab_regression:
    st.subheader("Prédire le prix d'un trajet de taxi")
    st.caption(f"Modèle déployé : {metadata['regression']['best_model']}")
    columns = metadata["regression"]
    defaults = columns["defaults"]
    categories = columns["categories"]

    with st.form("regression_form"):
        col1, col2 = st.columns(2)
        with col1:
            distance = st.number_input("Distance du trajet (km)", min_value=0.0, value=float(defaults["Trip_Distance_km"]), step=0.1)
            time_of_day = st.selectbox("Moment de la journée", categories["Time_of_Day"])
            day_of_week = st.selectbox("Jour", categories["Day_of_Week"])
            passenger_count = st.number_input("Nombre de passagers", min_value=1, value=int(round(defaults["Passenger_Count"])), step=1)
            traffic = st.selectbox("Conditions de circulation", categories["Traffic_Conditions"])
        with col2:
            weather = st.selectbox("Météo", categories["Weather"])
            base_fare = st.number_input("Tarif de base ($)", min_value=0.0, value=float(defaults["Base_Fare"]), step=0.01)
            per_km_rate = st.number_input("Tarif par km ($)", min_value=0.0, value=float(defaults["Per_Km_Rate"]), step=0.01)
            per_minute_rate = st.number_input("Tarif par minute ($)", min_value=0.0, value=float(defaults["Per_Minute_Rate"]), step=0.01)
            duration = st.number_input("Durée du trajet (minutes)", min_value=0.0, value=float(defaults["Trip_Duration_Minutes"]), step=0.1)
        predict_price = st.form_submit_button("Prédire le prix")

    if predict_price:
        input_df = pd.DataFrame([{
            "Trip_Distance_km": distance,
            "Time_of_Day": time_of_day,
            "Day_of_Week": day_of_week,
            "Passenger_Count": passenger_count,
            "Traffic_Conditions": traffic,
            "Weather": weather,
            "Base_Fare": base_fare,
            "Per_Km_Rate": per_km_rate,
            "Per_Minute_Rate": per_minute_rate,
            "Trip_Duration_Minutes": duration,
        }])
        prediction = regressor.predict(input_df)[0]
        st.success(f"Prix prédit : **{prediction:.2f} $**")
