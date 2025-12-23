from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_rd_clf():
    # Charger les données
    X_train, X_test, y_train, y_test = load_data()

    # Créer et entraîner le modèle
    rd_clf = RandomForestClassifier(
        criterion='entropy',
        max_depth=11,
        max_features='sqrt',   # correction ici
        min_samples_leaf=2,
        min_samples_split=3,
        n_estimators=130,
        random_state=42
    )

    rd_clf.fit(X_train, y_train)

    # Calcul de la précision
    rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

    # Affichage des résultats
    print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
    print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc}\n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
    print(f"Classification Report :- \n{classification_report(y_test, rd_clf.predict(X_test))}")

    # Retourner la variable pour l’importer ailleurs
    return rd_clf_acc , rd_clf