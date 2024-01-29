# Prédiction de la Qualité de l'Air à la Gare d'Auber

## Sommaire

- [Introduction](#introduction)
- [Technologies](#technologies)
- [Installation](#installation)
- [Structure et Contenu du Projet](#structure-et-contenu-du-projet)
- [Fonctionnement](#fonctionnement)
- [Utilisation](#utilisation)
- [Licence](#licence)
- [Contributeurs](#contributeurs)
- [Synthèse](#Synthèse)
- [Conclusion](#Conclusion)
  
## Introduction

Ce projet vise à analyser et prédire la qualité de l'air à la station Auber de la ligne A du RER, en utilisant des données fournies par la RATP. Il utilise l'apprentissage supervisé pour traiter et prédire la qualité de l'air.

## Technologies

- Python 3.11
- Pandas & Numpy pour le traitement des données
- Matplotlib pour la visualisation
- Scikit-Learn pour le machine learning
- Joblib pour la sauvegarde des modèles

## Installation

Pour installer les dépendances nécessaires, exécutez :

pip install -r requirements.txt


## Structure et Contenu du Projet

    dataset_aqi.csv : Données de qualité de l'air formatées avec AQI calculé.
    qualite-de-lair-mesuree-dans-la-station-auber.csv : Données brutes de la station Auber.
    Calcul_AQI.py : Script pour nettoyer et formater les données, et calculer l'AQI.
    Train_model_total.py : Script pour l'entraînement et l'évaluation du modèle RandomForest.
    graph.py : Script pour la visualisation des données d'AQI.
    random_forest_model.pkl : Modèle RandomForest entraîné et sauvegardé.
    aqi_predictions_advanced.csv : Prédictions de l'AQI pour les jours futurs.
    requirements.txt : Dépendances nécessaires pour le projet.

## Fonctionnement

Le projet fonctionne en plusieurs étapes :

    Nettoyage et Formatage des Données : Utilisation de Calcul_AQI.py pour préparer les données.
    Entraînement du Modèle : Train_model_total.py entraîne le modèle RandomForest avec les données nettoyées.
    Prédiction et Sauvegarde : Prédictions générées et sauvegardées dans aqi_predictions_advanced.csv.
    Visualisation : Création de graphiques pour analyser les tendances de l'AQI avec graph.py.

## Utilisation

Pour nettoyer les données et calculer l'AQI :

python Calcul_AQI.py


Pour entraîner le modèle et générer des prédictions :

python Train_model_total.py


Pour visualiser les résultats :

python graph.py

![Graph](https://github.com/MrBrabus75/Atelier-Big-Data/raw/main/1%20-%20TP%20Pr%C3%A9diction%20AQI/Graph.png)

## Synthèse 
Variabilité de l'AQI : Analyse de la variabilité journalière et des facteurs influents. 
Validité des Prédictions : Évaluation de la fiabilité des prédictions par rapport aux données réelles. 
Comparaison Temporelle : Analyse des tendances et des différences entre les mois. 
Implications pour la Santé Publique : Lien entre la qualité de l'air et les problèmes de santé publique. 
Importance des Données et Prédictions : Rôle des prévisions dans la planification environnementale et sanitaire.

## Conclusion 
L'analyse détaillée offre des insights cruciaux sur la qualité de l'air à la station Auber, soulignant les défis actuels et futurs. Ces informations sont essentielles pour la prise de décisions en matière de santé publique et de gestion environnementale.

