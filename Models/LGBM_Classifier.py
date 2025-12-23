from lightgbm import LGBMClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_lgbm():
    X_train, X_test, y_train, y_test = load_data()
    lgbm = LGBMClassifier(learning_rate=1)
    lgbm.fit(X_train, y_train)

    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

    print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
    print(f"Test Accuracy of LGBM Classifier is {lgbm_acc}\n")

    print(f"{confusion_matrix(y_test, lgbm.predict(X_test))}\n")
    print(classification_report(y_test, lgbm.predict(X_test)))

    return lgbm_acc , lgbm