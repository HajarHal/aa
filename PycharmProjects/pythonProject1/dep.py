import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# Titre de l'application
st.title('Prédiction des incendies')

# Description
st.write('Cette application utilise un modèle de machine learning pour prédire si un incendie va se produire en fonction des conditions météorologiques.')

# Formulaire pour saisir les paramètres
st.sidebar.header('Paramètres')
temperature = st.sidebar.slider('Température (°C)', min_value=-20.0, max_value=50.0, value=25.0, step=1.0)
humidity = st.sidebar.slider('Humidité (%)', min_value=0, max_value=100, value=50, step=1)
wind_speed = st.sidebar.slider('Vitesse du vent (km/h)', min_value=0, max_value=100, value=10, step=1)
brightness = st.sidebar.slider('Luminosité', min_value=0, max_value=500, value=250, step=10)
ndvi = st.sidebar.slider('Indice NDVI', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Création d'un dataframe avec les données entrées
input_data = pd.DataFrame({
    'brightness': [brightness],
    'temp': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    '500m 16 days NDVI': [ndvi]
})

# Chargement du modèle entraîné
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

# Normalisation des données d'entrée
scaler = MinMaxScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Prédiction
prediction = model.predict(input_data_scaled)
prediction_proba = model.predict_proba(input_data_scaled)

# Affichage du résultat
st.subheader('Résultat de la prédiction')
if prediction[0] == 1:
    st.write('Il y a une forte probabilité d\'incendie.')
else:
    st.write('Il y a une faible probabilité d\'incendie.')

st.write(f'Probabilité de l\'incendie: {prediction_proba[0][1]*100:.2f}%')

# Affichage des données d'entrée
st.subheader('Données d\'entrée')
st.write(input_data)
