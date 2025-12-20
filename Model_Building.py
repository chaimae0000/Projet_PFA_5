import pandas as pd
from sklearn.model_selection import train_test_split

cleaned_df= pd.read_csv('cleaned_kidney_disease.csv')
print(cleaned_df.head())
def load_data():
 ind_col = [col for col in cleaned_df.columns if col != 'class']
 dep_col = 'class'

 X = cleaned_df[ind_col]
 y = cleaned_df[dep_col]

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

 return X_train, X_test, y_train, y_test