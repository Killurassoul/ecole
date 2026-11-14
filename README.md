# Projet Machine Learning

## Notebooks à déposer

- `Classification_fleurs_iris.ipynb` : classification des fleurs Iris.
- `Clustering_clients_KMeans_CAH.ipynb` : segmentation des clients avec K-Means et CAH.
- `Clustering_ble_KMeans_CAH.ipynb` : segmentation des graines de blé avec K-Means et CAH.

Les deux fichiers de données utilisés par les notebooks de clustering sont dans `data sets/`.

## Lancer l'application

Depuis ce dossier :

```powershell
cd streamlit_app
streamlit run app.py
```

L'application regroupe les trois travaux. Dans les deux onglets de clustering, on peut choisir K-Means ou CAH ainsi que le nombre de groupes.

