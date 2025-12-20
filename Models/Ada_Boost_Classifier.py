from sklearn.ensemble import AdaBoostClassifier
from Models.Decision_Tree_Classifier import train_dtc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_ada_boost():
    # Chargement   des données
    X_train, X_test, y_train, y_test = load_data()
    dtc = train_dtc()[1]  # Obtenir le modèle DTC entraîné
    ada = AdaBoostClassifier(base_estimator = dtc)
    ada.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of ada boost

    ada_acc = accuracy_score(y_test, ada.predict(X_test))

    print(f"Training Accuracy of Ada Boost Classifier is {accuracy_score(y_train, ada.predict(X_train))}")
    print(f"Test Accuracy of Ada Boost Classifier is {ada_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, ada.predict(X_test))}")
    # Retourner le score test et le modèle
    return ada_acc
