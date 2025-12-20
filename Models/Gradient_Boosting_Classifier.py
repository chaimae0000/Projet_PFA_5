from sklearn.ensemble import GradientBoostingClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_gradient_boosting():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()

    # Création et entraînement du modèle
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)

    # Évaluation
    train_acc = accuracy_score(y_train, gb.predict(X_train))
    gb_acc = accuracy_score(y_test, gb.predict(X_test))

    print(f"Training Accuracy of Gradient Boosting Classifier is {train_acc}")
    print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")

    # Retourner le score test et le modèle
    return gb_acc