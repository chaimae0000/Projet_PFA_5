import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(return_preprocessor=False):
    df = pd.read_csv("data/cleaned_kidney_disease.csv")  # adapte le chemin
    y = df["class"]
    X = df.drop(columns=["class"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  # optionnel, tu peux enlever
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessing uniquement sur train
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    if return_preprocessor:
        return X_train_t, X_test_t, y_train, y_test, preprocessor, X.columns.tolist()

    return X_train_t, X_test_t, y_train, y_test
