from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data

def train_knn():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()
    
    # Création et entraînement du modèle
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    # Calcul de la précision
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    
    # Affichage des résultats
    print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
    print(f"Test Accuracy of KNN is {knn_acc}\n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")
    
    # Retourner la variable pour l’importer ailleurs
    return knn_acc