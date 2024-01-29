import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

# Charger le dataset
data = pd.read_csv('C:\\Users\\PC TOUR\\Desktop\\ML\\houses_to_rent_1.csv')

# Exploration basique
print(data.head())

# Identifier les colonnes catégorielles et numériques
categorical_columns = ['animal', 'furniture']
numerical_columns = data.columns.drop(['rent amount', 'animal', 'furniture'])

# Prétraitement avec OneHotEncoder pour les catégorielles et StandardScaler pour les numériques
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Séparation des features et de la cible
X = data.drop('rent amount', axis=1)
y = data['rent amount']

# Application du prétraitement à l'ensemble du dataset
X_transformed = preprocessor.fit_transform(X)

# Utilisation de la validation croisée pour évaluer le modèle
model = LinearRegression()
scores = cross_val_score(model, X_transformed, y, cv=5, scoring='neg_mean_squared_error')

# Calcul de la moyenne des erreurs quadratiques moyennes pour les différents plis
mse_scores = -scores
print(f"MSE moyen lors de la validation croisée : {np.mean(mse_scores)}")

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Entraînement du modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Évaluation du modèle sur l'ensemble de test
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Erreur quadratique moyenne sur l'ensemble de test: {mse}")

# Sauvegarde du modèle et du preprocessor
joblib.dump(model, 'C:\\Users\\PC TOUR\\Desktop\\ML\\house_rent_model.pkl')
joblib.dump(preprocessor, 'C:\\Users\\PC TOUR\\Desktop\\ML\\preprocessor.pkl')
