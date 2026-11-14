import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data sets"

st.set_page_config(page_title="TP Machine Learning", layout="wide")


@st.cache_resource
def load_classifier():
    return joblib.load(ARTIFACTS_DIR / "best_classifier.joblib")


@st.cache_data
def load_data(file_name):
    return pd.read_csv(DATA_DIR / file_name)


classifier = load_classifier()
metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))

st.title("TP Machine Learning")
st.write("Classification Iris et clustering avec K-Means et CAH.")
tab_iris, tab_clients, tab_ble = st.tabs(
    ["Iris", "Clients", "Blé"]
)

with tab_iris:
    st.subheader("Prédire l'espèce d'une fleur Iris")
    st.caption(f"Modèle du notebook : {metadata['classification']['best_model']}")
    st.write("Saisis les mesures de la fleur en centimètres.")
    with st.form("iris_form"):
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Longueur du sépale", min_value=0.0, value=5.8, step=0.1)
            sepal_width = st.number_input("Largeur du sépale", min_value=0.0, value=3.0, step=0.1)
        with col2:
            petal_length = st.number_input("Longueur du pétale", min_value=0.0, value=4.0, step=0.1)
            petal_width = st.number_input("Largeur du pétale", min_value=0.0, value=1.2, step=0.1)
        classify = st.form_submit_button("Prédire l'espèce")

    if classify:
        input_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=metadata["classification"]["features"],
        )
        prediction = classifier.predict(input_df)[0]
        st.success(f"Espèce prédite : **{prediction}**")


def clustering_tab(title, file_name, subject, default_clusters):
    st.subheader(title)
    st.write("Choisis une méthode et le nombre de groupes.")
    df = load_data(file_name).dropna().copy()
    x = normalize(df.values)

    method = st.radio(
        "Méthode de clustering",
        ["K-Means", "CAH"],
        horizontal=True,
        key=f"method_{subject}",
    )
    n_clusters = st.slider(
        "Nombre de groupes",
        min_value=2,
        max_value=8,
        value=default_clusters,
        key=f"clusters_{subject}",
    )

    if method == "K-Means":
        model = KMeans(
            n_clusters=n_clusters,
            init="random",
            max_iter=500,
            random_state=42,
            n_init=10,
        )
        partition = model.fit_predict(x)
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="average",
        )
        partition = model.fit_predict(x)

    score = silhouette_score(x, partition)
    st.metric("Score silhouette", f"{score:.3f}")

    counts = pd.Series(partition).value_counts().sort_index()
    results = df.copy()
    results["Groupe"] = partition
    counts_df = pd.DataFrame(
        {"Groupe": counts.index.astype(str), f"Nombre de {subject}": counts.values}
    )
    fig_counts = px.bar(
        counts_df,
        x="Groupe",
        y=f"Nombre de {subject}",
        color="Groupe",
        text_auto=True,
        title=f"Nombre de {subject} par groupe",
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    pca_model = PCA(n_components=2)
    components = pca_model.fit_transform(x)
    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["Groupe"] = partition.astype(str)
    fig_scatter = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Groupe",
        title=f"Projection des groupes — {method}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        f"L'ACP représente les données en deux dimensions pour le graphique "
        f"({pca_model.explained_variance_ratio_.sum() * 100:.1f}% de variance affichée)."
    )
    with st.expander("Voir les données avec le groupe attribué"):
        st.dataframe(results, use_container_width=True)


with tab_clients:
    clustering_tab(
        "Segmentation des clients télécom",
        "Customer_dataset.csv",
        "clients",
        3,
    )

with tab_ble:
    clustering_tab(
        "Segmentation des graines de blé",
        "wheat_seeds_dataset.csv",
        "graines",
        3,
    )

