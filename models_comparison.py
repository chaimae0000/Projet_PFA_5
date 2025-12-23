import os
import sys
import warnings
import pickle
import pandas as pd

from Models.Ada_Boost_Classifier import train_ada_boost
from Models.Cat_Boost_Classifier import train_catboost
from Models.Decision_Tree_Classifier import train_dtc
from Models.Extra_Trees_Classifier import train_extra_trees
from Models.Gradient_Boosting_Classifier import train_gradient_boosting
from Models.KNN import train_knn
from Models.LGBM_Classifier import train_lgbm
from Models.Random_Forest_Classifier import train_rd_clf
from Models.SGB import train_stochastic_gradient_boosting
from Models.XgBoost import train_xgb

warnings.filterwarnings("ignore")

def silence(func, *args, **kwargs):
    """Exécute une fonction en silence (sans prints)."""
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = stdout, stderr
    return result

def ensure_tuple(res, model_name: str):
    """
    S'assure que la fonction retourne bien (score, model).
    Si ce n'est pas le cas, lève une erreur claire.
    """
    if isinstance(res, tuple) and len(res) == 2:
        return res
    raise ValueError(
        f"{model_name} doit retourner (score, model). "
        f"Actuellement la fonction retourne: {type(res)} => {res}"
    )

def main():
    # Chaque train_* doit retourner (acc, model)
    results = []

    results.append(("KNN", *ensure_tuple(silence(train_knn), "train_knn")))
    results.append(("Decision Tree", *ensure_tuple(silence(train_dtc), "train_dtc")))
    results.append(("Random Forest", *ensure_tuple(silence(train_rd_clf), "train_rd_clf")))
    results.append(("AdaBoost", *ensure_tuple(silence(train_ada_boost), "train_ada_boost")))
    results.append(("Gradient Boosting", *ensure_tuple(silence(train_gradient_boosting), "train_gradient_boosting")))
    results.append(("Stochastic GB", *ensure_tuple(silence(train_stochastic_gradient_boosting), "train_stochastic_gradient_boosting")))
    results.append(("XGBoost", *ensure_tuple(silence(train_xgb), "train_xgb")))
    results.append(("CatBoost", *ensure_tuple(silence(train_catboost), "train_catboost")))
    results.append(("Extra Trees", *ensure_tuple(silence(train_extra_trees), "train_extra_trees")))

    # Tableau scores
    df = pd.DataFrame(
        [{"Model": name, "Score": score} for (name, score, _model) in results]
    ).sort_values(by="Score", ascending=False)

    print(df)

    # Meilleur modèle
    best_name, best_score, best_model = max(results, key=lambda x: x[1])
    print(f"\nBest model: {best_name} | Score={best_score}")

    # Sauvegarde dans artifacts/bestmodel.pkl (chemin robuste)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    best_path = os.path.join(artifacts_dir, "bestmodel.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Saved best model to: {best_path}")

if __name__ == "__main__":
    main()
