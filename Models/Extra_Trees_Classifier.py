from sklearn.ensemble import ExtraTreesClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_extra_trees():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()

    # Création et entraînement du modèle
    etc = ExtraTreesClassifier()
    etc.fit(X_train, y_train)

    # Évaluation
    train_acc = accuracy_score(y_train, etc.predict(X_train))
    etc_acc = accuracy_score(y_test, etc.predict(X_test))

    print(f"Training Accuracy of Extra Trees Classifier is {train_acc}")
    print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")

    # Retourner le score test et le modèle
    return etc_acc , etc