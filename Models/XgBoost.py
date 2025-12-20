from xgboost import XGBClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model_Building import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_xgb():
    X_train, X_test, y_train, y_test = load_data()
    
    xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.5, max_depth=5, n_estimators=150)
    xgb.fit(X_train, y_train)
    
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    
    print(f"Training Accuracy of XgBoost is {accuracy_score(y_train, xgb.predict(X_train))}")
    print(f"Test Accuracy of XgBoost is {xgb_acc} \n")
    
    print(f"Confusion Matrix :- \n{confusion_matrix(y_test, xgb.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(y_test, xgb.predict(X_test))}")
    
    return xgb_acc