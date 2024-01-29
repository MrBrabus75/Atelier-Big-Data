import pandas as pd
import joblib

# Charger le modèle et le preprocessor sauvegardés
model = joblib.load('C:\\Users\\PC TOUR\\Desktop\\ML\\house_rent_model.pkl')
preprocessor = joblib.load('C:\\Users\\PC TOUR\\Desktop\\ML\\preprocessor.pkl')

# Charger les nouvelles données
new_data = pd.read_csv('C:\\Users\\PC TOUR\\Desktop\\ML\\new_data.csv')

# Application du prétraitement aux nouvelles données
new_data_transformed = preprocessor.transform(new_data)

# Faire des prédictions
predictions = model.predict(new_data_transformed)

# Ajouter les prédictions au dataset et le sauvegarder
new_data['predicted_rent_amount'] = predictions
new_data.to_csv('C:\\Users\\PC TOUR\\Desktop\\ML\\new_data_with_predictions.csv', index=False)
