from Models.Ada_Boost_Classifier import train_ada_boost
from Models.Cat_Boost_Classifier import train_catboost
from Models.Decision_Tree_Classifier import train_dtc
from Models.Extra_Trees_Classifier import train_extra_trees
from Models.Gradient_Boosting_Classifier import train_gradient_boosting
from Models.KNN import train_knn
from Models.LGBM_Classifier import train_lgbm
from Models.Random_Forest_Classifier import train_rd_clf
from Models.SGB import train_stochastic_gradient_boosting
from Models.XgBoost import train_xgb
import pandas as pd
import warnings
import sys
import os
warnings.filterwarnings('ignore')
def silence(func, *args, **kwargs):
    """Exécute une fonction en silence (sans afficher quoi que ce soit)."""
    stdout = sys.stdout
    sys.stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    result = func(*args, **kwargs)
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = stdout
    sys.stderr = sys.stderr
    return result

knn_acc = silence(train_knn)
dtc_acc = silence(train_dtc)
rd_clf_acc = silence(train_rd_clf)
ada_acc = silence(train_ada_boost)
gb_acc = silence(train_gradient_boosting)
sgb_acc = silence(train_stochastic_gradient_boosting)
xgb_acc = silence(train_xgb)
cat_acc = silence(train_catboost)
etc_acc = silence(train_extra_trees)


models = pd.DataFrame({
    'Model' : [ 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier','Gradient Boosting Classifier', 'Stochastic Gradient Boosting', 'XgBoost', 'Cat Boost', 'Extra Trees Classifier'],
    'Score' : [knn_acc, dtc_acc, rd_clf_acc, ada_acc, gb_acc, sgb_acc, xgb_acc, cat_acc, etc_acc]
})


models = models.sort_values(by='Score', ascending=False)

print(models)