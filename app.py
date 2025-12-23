import streamlit as st
import pandas as pd
import pickle
import os

BEST_PATH = os.path.join("artifacts", "bestmodel.pkl")

st.set_page_config(page_title="CKD Predictor", layout="wide")
st.title("Chronic Kidney Disease Predictor (CSV Upload)")

@st.cache_resource
def load_artifact():
    with open(BEST_PATH, "rb") as f:
        return pickle.load(f)

artifact = load_artifact()
model = artifact["model"]
preprocessor = artifact["preprocessor"]
feature_columns = artifact["feature_columns"]

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Aperçu du fichier")
    st.dataframe(df.head(20))

    # Supprimer target si elle existe dans le CSV
    if "class" in df.columns:
        df = df.drop(columns=["class"])

    # Vérifier colonnes attendues
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans ton CSV: {missing}")
        st.stop()

    # Garder ordre exact
    df = df[feature_columns]

    # Transformer + prédire
    X = preprocessor.transform(df)
    preds = model.predict(X)

    out = df.copy()
    out["prediction"] = preds

    st.subheader("Résultats")
    st.dataframe(out.head(50))

    st.download_button(
        "Télécharger le CSV avec prédictions",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )
