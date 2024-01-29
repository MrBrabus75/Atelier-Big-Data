
### Modélisation Prédictive des Loyers : Application du Machine Learning et de l'IA

Ce projet vise à développer un modèle prédictif capable d'estimer le montant du loyer des maisons en fonction de diverses caractéristiques immobilières. L'objectif est de créer un modèle efficace et de faciliter les prédictions sur de nouvelles données en utilisant des techniques de Machine Learning et d'Intelligence Artificielle (IA). Pour ce faire, nous avons mis en place plusieurs étapes clés et utilisé différents fichiers.
Configuration Requise


## Requirements

 Pour exécuter ces scripts, assurez-vous d'avoir les bibliothèques suivantes installées en utilisant le fichier requirements.txt :

joblib
pandas
scikit-learn
numpy


## Description des Fichiers

# creation_entrainement_sauvegarde.py

Ce script joue un rôle crucial dans la construction du modèle prédictif. Voici les étapes détaillées :

    Chargement et Exploration des Données : Nous commençons par charger le dataset houses_to_rent_1.csv, qui contient des informations sur les propriétés et leurs loyers. Nous explorons ces données pour en avoir une première compréhension.

    Prétraitement des Données : Les données sont préparées pour l'analyse. Nous utilisons des techniques telles que la normalisation et l'encodage OneHot pour gérer les variables numériques et catégorielles, ce qui permet au modèle de les comprendre.

    Modélisation et Évaluation : Un modèle de régression linéaire est construit pour quantifier la relation entre les caractéristiques des propriétés (comme la taille, le nombre de pièces, la présence d'animaux, le mobilier, etc.) et le montant du loyer. Nous évaluons la performance du modèle en utilisant la validation croisée pour garantir sa fiabilité.

    Entraînement et Test du Modèle : Le modèle est entraîné sur une partie des données et testé sur une autre pour évaluer sa capacité à généraliser les prédictions.

    Sauvegarde du Modèle : Enfin, nous sauvegardons le modèle dans le fichier house_rent_model.pkl pour une utilisation future.


# utilisation_modele.py

Ce script est destiné à l'utilisation du modèle sauvegardé pour faire des prédictions sur de nouvelles données. Voici les étapes :

    Chargement du Modèle Sauvegardé et du Préprocesseur : Le modèle préalablement entraîné et le préprocesseur utilisé pour traiter les données sont chargés depuis les fichiers sauvegardés.

    Chargement des Nouvelles Données : Les données immobilières récentes sont chargées depuis le fichier new_data.csv.

    Application du Prétraitement aux Nouvelles Données : Les mêmes transformations de prétraitement appliquées aux données d'entraînement sont également appliquées aux nouvelles données.

    Faire des Prédictions sur le Montant du Loyer : Le modèle est utilisé pour faire des prédictions sur le montant du loyer en se basant sur les caractéristiques des nouvelles données.

    Sauvegarde des Résultats : Les prédictions sont ajoutées aux données d'entrée, puis le résultat est sauvegardé dans le fichier new_data_with_predictions.csv.


# Datasets Utilisés

    houses_to_rent_1.csv : Ce dataset initial contient des informations détaillées sur les propriétés et les loyers. Il est utilisé pour l'entraînement du modèle.

    new_data.csv : Ce fichier contient de nouvelles données immobilières qui servent à tester la capacité du modèle à faire des prédictions sur des données récentes.

    new_data_with_predictions.csv : Ce fichier final inclut les prédictions de loyer effectuées par le modèle, illustrant ainsi l'application pratique du modèle.

## Conclusion

Ce projet démontre un processus complet de Machine Learning, de la préparation des données à la prédiction en passant par l'entraînement du modèle. L'objectif est de fournir un outil fiable pour estimer le loyer des propriétés, ce qui peut être un atout précieux pour les professionnels de l'immobilier. En utilisant l'IA et le ML, nous sommes en mesure de résoudre des problèmes concrets et de prendre des décisions éclairées dans le domaine de l'immobilier.
