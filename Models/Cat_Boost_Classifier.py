from catboost import CatBoostClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_catboost():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()

    # Création et entraînement du modèle
    cat = CatBoostClassifier(iterations=10, verbose=0)
    cat.fit(X_train, y_train)

    # Évaluation
    train_acc = accuracy_score(y_train, cat.predict(X_train))
    cat_acc = accuracy_score(y_test, cat.predict(X_test))

    print(f"Training Accuracy of CatBoost Classifier is {train_acc}")
    print(f"Test Accuracy of CatBoost Classifier is {cat_acc} \n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, cat.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, cat.predict(X_test))}")

    # Retourner le score test et le modèle
    return cat_acc , cat