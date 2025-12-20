from sklearn.tree import DecisionTreeClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def train_dtc():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()
    
    # Création et entraînement du modèle
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    
    # Calcul de la précision
    dtc_acc = accuracy_score(y_test, dtc.predict(X_test))
    
    # Affichage des résultats
    print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
    print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc}\n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")
    
    # Retourner la variable pour l’importer ailleurs
    return dtc_acc , dtc