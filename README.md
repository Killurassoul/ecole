# Projet Machine Learning

## Fichiers à déposer

- `Classification_fleurs_iris.ipynb` : classification des fleurs Iris.
- `Regression_prix_taxi.ipynb` : régression du prix d'un trajet de taxi.

Les autres modèles sont conservés dans les notebooks pour la comparaison. Les meilleurs modèles seulement sont enregistrés pour le déploiement :

- Classification : `Logistic Regression` → `streamlit_app/artifacts/best_classifier.joblib`
- Régression : `Random Forest` → `streamlit_app/artifacts/best_regressor.joblib`

## Lancer l'application

Depuis ce dossier :

```powershell
cd streamlit_app
streamlit run app.py
```

L'application contient un onglet Classification et un onglet Régression.
