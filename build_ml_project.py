"""Build the two notebooks, train the selected models and save deployment files.

This script is intentionally kept with the project so that the work is reproducible:
    python build_ml_project.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "streamlit_app" / "artifacts"
DATA_DIR = BASE_DIR / "data"

RANDOM_STATE = 42


def classification_workflow() -> tuple[pd.DataFrame, str, dict]:
    """Train and compare four classifiers on the Iris dataset."""
    iris = load_iris(as_frame=True)
    classification_df = iris.frame.copy()
    classification_df["species"] = classification_df["target"].map(
        dict(enumerate(iris.target_names))
    )
    classification_df = classification_df.drop(columns="target")
    classification_df.to_csv(DATA_DIR / "iris_classification.csv", index=False)

    X = classification_df.drop(columns="species")
    y = classification_df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    classifiers = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [None, 3, 5, 8], "model__min_samples_leaf": [1, 2, 4]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 3, 5],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }

    fitted_models = {}
    rows = []
    for name, (model, parameters) in classifiers.items():
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", model)]
        )
        search = GridSearchCV(
            pipeline, parameters, cv=5, scoring="f1_weighted", n_jobs=-1
        )
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        fitted_models[name] = search.best_estimator_
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
                "F1-score": f1_score(y_test, prediction, average="weighted", zero_division=0),
                "Best parameters": search.best_params_,
            }
        )

    results = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, ARTIFACTS_DIR / "best_classifier.joblib")
    return results, best_name, {"features": list(X.columns), "classes": sorted(y.unique())}


def regression_workflow() -> tuple[pd.DataFrame, str, dict]:
    """Train and compare four regressors on the supplied taxi dataset."""
    data_path = BASE_DIR / "taxi_trip_pricing.csv"
    taxi_df = pd.read_csv(data_path)
    target = "Trip_Price"
    taxi_df = taxi_df.dropna(subset=[target]).copy()

    X = taxi_df.drop(columns=target)
    y = taxi_df[target]
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    regressors = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"model__max_depth": [None, 4, 6, 8, 10], "model__min_samples_leaf": [1, 2, 4]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }

    fitted_models = {}
    rows = []
    for name, (model, parameters) in regressors.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        search = GridSearchCV(
            pipeline, parameters, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        fitted_models[name] = search.best_estimator_
        rows.append(
            {
                "Model": name,
                "R2": r2_score(y_test, prediction),
                "MAE": mean_absolute_error(y_test, prediction),
                "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                "Best parameters": search.best_params_,
            }
        )

    results = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, ARTIFACTS_DIR / "best_regressor.joblib")
    return results, best_name, {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categories": {column: sorted(taxi_df[column].dropna().unique().tolist()) for column in categorical_features},
        "defaults": {column: float(taxi_df[column].median()) for column in numeric_features},
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def build_classification_notebook() -> None:
    write_notebook(
        BASE_DIR / "Classification_fleurs_iris.ipynb",
        [
            markdown("# Notebook 1 — Classification\n\n## Prédiction de l'espèce d'une fleur Iris\n\n**Objectif :** comparer plusieurs algorithmes de classification et déployer uniquement le plus performant."),
            markdown("### 1. Importation des bibliothèques et chargement des données"),
            code("""import pandas as pd\nimport joblib\nfrom pathlib import Path\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n\nRANDOM_STATE = 42\nBASE_DIR = Path.cwd()\nARTIFACTS_DIR = BASE_DIR / 'streamlit_app' / 'artifacts'\n\niris = load_iris(as_frame=True)\ndf = iris.frame.copy()\ndf['species'] = df['target'].map(dict(enumerate(iris.target_names)))\ndf = df.drop(columns='target')\ndf.head()"""),
            markdown("### 2. Préparation des données\nLa variable cible est `species`. La stratification conserve la même proportion de chaque espèce dans les jeux d'entraînement et de test."),
            code("""X = df.drop(columns='species')\ny = df['species']\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE\n)\nprint(X_train.shape, X_test.shape)"""),
            markdown("### 3. Entraînement et comparaison\nLes modèles testés restent dans ce notebook. La métrique de sélection est le **F1-score pondéré**, car il prend en compte precision et recall."),
            code("""models = {\n    'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), {'model__C': [0.1, 1, 10]}),\n    'K-Nearest Neighbors': (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}),\n    'Decision Tree': (DecisionTreeClassifier(random_state=RANDOM_STATE), {'model__max_depth': [None, 3, 5, 8], 'model__min_samples_leaf': [1, 2, 4]}),\n    'Random Forest': (RandomForestClassifier(random_state=RANDOM_STATE), {'model__n_estimators': [100, 200], 'model__max_depth': [None, 3, 5], 'model__min_samples_leaf': [1, 2]})\n}\n\nfitted_models, rows = {}, []\nfor name, (model, params) in models.items():\n    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])\n    grid = GridSearchCV(pipeline, params, cv=5, scoring='f1_weighted', n_jobs=-1)\n    grid.fit(X_train, y_train)\n    prediction = grid.predict(X_test)\n    fitted_models[name] = grid.best_estimator_\n    rows.append({\n        'Model': name,\n        'Accuracy': accuracy_score(y_test, prediction),\n        'Precision': precision_score(y_test, prediction, average='weighted', zero_division=0),\n        'Recall': recall_score(y_test, prediction, average='weighted', zero_division=0),\n        'F1-score': f1_score(y_test, prediction, average='weighted', zero_division=0),\n        'Best parameters': grid.best_params_\n    })\n\nresults = pd.DataFrame(rows).sort_values('F1-score', ascending=False).reset_index(drop=True)\nresults"""),
            markdown("### 4. Choix et sauvegarde du meilleur modèle\nSeul le meilleur modèle est enregistré pour l'application Streamlit. Les autres modèles restent dans le notebook comme preuve de comparaison."),
            code("""best_model_name = results.loc[0, 'Model']\nbest_classifier = fitted_models[best_model_name]\njoblib.dump(best_classifier, ARTIFACTS_DIR / 'best_classifier.joblib')\nprint(f'Modèle retenu : {best_model_name}')"""),
            markdown("### Conclusion\nLe modèle retenu est celui ayant le meilleur F1-score sur les données de test. Il sera utilisé dans l'onglet **Classification** de l'application Streamlit."),
        ],
    )


def build_regression_notebook() -> None:
    write_notebook(
        BASE_DIR / "Regression_prix_taxi.ipynb",
        [
            markdown("# Notebook 2 — Régression\n\n## Prédiction du prix d'un trajet de taxi\n\n**Objectif :** comparer plusieurs modèles de régression et déployer uniquement le plus performant."),
            markdown("### 1. Importation et lecture des données"),
            code("""import pandas as pd\nimport numpy as np\nimport joblib\nfrom pathlib import Path\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.linear_model import LinearRegression, Ridge\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n\nRANDOM_STATE = 42\nBASE_DIR = Path.cwd()\nARTIFACTS_DIR = BASE_DIR / 'streamlit_app' / 'artifacts'\ndf = pd.read_csv(BASE_DIR / 'taxi_trip_pricing.csv')\ndf.head()"""),
            markdown("### 2. Nettoyage et préparation\nLa cible est `Trip_Price`, une valeur numérique continue : c'est donc un problème de régression. Les valeurs manquantes sont traitées dans le pipeline afin d'éviter les fuites de données."),
            code("""target = 'Trip_Price'\ndf = df.dropna(subset=[target]).copy()\nX = df.drop(columns=target)\ny = df[target]\n\nnumeric_features = X.select_dtypes(include=np.number).columns.tolist()\ncategorical_features = X.select_dtypes(exclude=np.number).columns.tolist()\n\npreprocessor = ColumnTransformer([\n    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),\n    ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)\n])\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)\nprint('Variables numériques :', numeric_features)\nprint('Variables catégorielles :', categorical_features)"""),
            markdown("### 3. Entraînement et comparaison\nLes quatre modèles sont conservés dans le notebook. La sélection du meilleur modèle se fait avec le **RMSE** : plus il est faible, meilleur est le modèle."),
            code("""models = {\n    'Linear Regression': (LinearRegression(), {}),\n    'Ridge': (Ridge(), {'model__alpha': [0.1, 1, 10, 100]}),\n    'Decision Tree': (DecisionTreeRegressor(random_state=RANDOM_STATE), {'model__max_depth': [None, 4, 6, 8, 10], 'model__min_samples_leaf': [1, 2, 4]}),\n    'Random Forest': (RandomForestRegressor(random_state=RANDOM_STATE), {'model__n_estimators': [100, 200], 'model__max_depth': [None, 6, 10], 'model__min_samples_leaf': [1, 2, 4]})\n}\n\nfitted_models, rows = {}, []\nfor name, (model, params) in models.items():\n    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])\n    grid = GridSearchCV(pipeline, params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)\n    grid.fit(X_train, y_train)\n    prediction = grid.predict(X_test)\n    fitted_models[name] = grid.best_estimator_\n    rows.append({\n        'Model': name,\n        'R2': r2_score(y_test, prediction),\n        'MAE': mean_absolute_error(y_test, prediction),\n        'RMSE': mean_squared_error(y_test, prediction) ** 0.5,\n        'Best parameters': grid.best_params_\n    })\n\nresults = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)\nresults"""),
            markdown("### 4. Choix et sauvegarde du meilleur modèle\nLe pipeline complet est enregistré : il contient le prétraitement et le modèle. L'application peut donc prédire avec de nouvelles données sans reconstruire manuellement les encodeurs et le scaler."),
            code("""best_model_name = results.loc[0, 'Model']\nbest_regressor = fitted_models[best_model_name]\njoblib.dump(best_regressor, ARTIFACTS_DIR / 'best_regressor.joblib')\nprint(f'Modèle retenu : {best_model_name}')"""),
            markdown("### Conclusion\nLe meilleur modèle selon le RMSE est déployé dans l'onglet **Régression** de l'application Streamlit. Les autres modèles restent disponibles dans ce notebook."),
        ],
    )


def write_streamlit_app() -> None:
    app_dir = BASE_DIR / "streamlit_app"
    app_dir.mkdir(exist_ok=True)
    app_source = '''import json
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
'''
    (app_dir / "app.py").write_text(app_source, encoding="utf-8")
    (app_dir / "requirements.txt").write_text(
        "streamlit\npandas\nscikit-learn\njoblib\nnumpy\n", encoding="utf-8"
    )


def write_readme(classifier_name: str, regressor_name: str) -> None:
    readme = f"""# Projet Machine Learning\n\n## Fichiers à déposer\n\n- `Classification_fleurs_iris.ipynb` : classification des fleurs Iris.\n- `Regression_prix_taxi.ipynb` : régression du prix d'un trajet de taxi.\n\nLes autres modèles sont conservés dans les notebooks pour la comparaison. Les meilleurs modèles seulement sont enregistrés pour le déploiement :\n\n- Classification : `{classifier_name}` → `streamlit_app/artifacts/best_classifier.joblib`\n- Régression : `{regressor_name}` → `streamlit_app/artifacts/best_regressor.joblib`\n\n## Lancer l'application\n\nDepuis ce dossier :\n\n```powershell\ncd streamlit_app\nstreamlit run app.py\n```\n\nL'application contient un onglet Classification et un onglet Régression.\n"""
    (BASE_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    classification_results, classifier_name, classification_metadata = classification_workflow()
    regression_results, regressor_name, regression_metadata = regression_workflow()

    metadata = {
        "classification": {"best_model": classifier_name, **classification_metadata},
        "regression": {"best_model": regressor_name, **regression_metadata},
        "results": {
            "classification": classification_results.to_dict(orient="records"),
            "regression": regression_results.to_dict(orient="records"),
        },
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    build_classification_notebook()
    build_regression_notebook()
    write_streamlit_app()
    write_readme(classifier_name, regressor_name)

    print("Projet créé avec succès.")
    print(f"Meilleur classifieur : {classifier_name}")
    print(f"Meilleur régresseur : {regressor_name}")
    print("Notebooks : Classification_fleurs_iris.ipynb et Regression_prix_taxi.ipynb")


if __name__ == "__main__":
    main()
