import pandas as pd
import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import joblib

# Configuration de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les données
file_path = 'dataset_aqi.csv'
try:
    data = pd.read_csv(file_path, sep=';')
    logging.info("Fichier de données chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du fichier de données : {e}")
    exit()

# Analyse approfondie des données et feature engineering
try:
    # Convertir en datetime et trier
    data['DATE'] = pd.to_datetime(data['DATE'])
    data.sort_values('DATE', inplace=True)

    # Sélectionner les caractéristiques et la cible
    features = data[['NO', 'NO2', 'PM10', 'PM2.5', 'CO2', 'TEMP', 'HUMI']]
    target = data['AQI']

    # Normalisation des caractéristiques
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    # Séparer en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    logging.info("Analyse des données et feature engineering terminés.")
except Exception as e:
    logging.error(f"Erreur lors de l'analyse des données : {e}")
    exit()

# Modèle de machine learning avancé
try:
    # Utilisation d'un RandomForest pour la sélection de caractéristiques
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X_train, y_train)
    X_train_sel = sel.transform(X_train)
    X_test_sel = sel.transform(X_test)

    # Entraînement d'un modèle RandomForest
    clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    clf.fit(X_train_sel, y_train)

    # Évaluation du modèle
    y_pred = clf.predict(X_test_sel)
    logging.info(f"Rapport de classification :\n{classification_report(y_test, y_pred)}")
    logging.info(f"Matrice de confusion :\n{confusion_matrix(y_test, y_pred)}")

    logging.info("Entraînement et évaluation du modèle RandomForest terminés.")
except Exception as e:
    logging.error(f"Erreur lors de l'entraînement du modèle avancé : {e}")
    exit()


# Fonction pour déterminer le statut en fonction de l'AQI
def determine_statut(aqi):
    if aqi <= 50:
        return 'Bon'
    elif aqi <= 150:
        return 'Modéré'
    elif aqi <= 300:
        return 'Dangereux'
    else:
        return 'Très dangereux'

# Prédiction des valeurs AQI pour les 160 prochains jours
try:
    # Préparer les données pour la prédiction
    last_days = feature_scaler.transform(features[-160:])  # Normalisation
    last_days_selected = sel.transform(last_days)  # Sélection de caractéristiques

    future_predictions = clf.predict(last_days_selected)

    # Créer un DataFrame pour les prédictions
    future_dates = pd.date_range(start=data['DATE'].iloc[-1] + pd.Timedelta(days=1), periods=160)
    statuts = [determine_statut(aqi) for aqi in future_predictions]
    predicted_data = pd.DataFrame({'DATE': future_dates, 'AQI': future_predictions, 'STATUT': statuts})
    predicted_data.to_csv('aqi_predictions_advanced.csv', index=False)
    logging.info("Prédictions enregistrées dans 'aqi_predictions_advanced.csv'.")
except Exception as e:
    logging.error(f"Erreur lors de la prédiction des valeurs AQI : {e}")
    exit()

# Sauvegarder le modèle
try:
    joblib.dump(clf, 'random_forest_model.pkl')
    logging.info("Modèle RandomForest sauvegardé sous 'random_forest_model.pkl'.")
except Exception as e:
    logging.error(f"Erreur lors de la sauvegarde du modèle : {e}")
