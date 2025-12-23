from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier

def train_stochastic_gradient_boosting():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()

    # Création et entraînement du modèle
    sgb = GradientBoostingClassifier(
        max_depth=4,
        subsample=0.90,
        max_features=0.75,
        n_estimators=200,
        random_state=42
    )
    sgb.fit(X_train, y_train)

    # Évaluation
    train_acc = accuracy_score(y_train, sgb.predict(X_train))
    sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

    print(f"Training Accuracy of Stochastic Gradient Boosting is {train_acc}")
    print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")

    # Retourner le score test et le modèle
    return sgb_acc , sgb