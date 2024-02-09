import numpy as np 
import pandas as pd 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv('D:\Major_Project\Dataset\clean_dataset.csv')

X = df.drop('Approved', axis = 1)
y = df['Approved']

X = pd.get_dummies(X)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify = y)

model = RandomForestClassifier(random_state=23, class_weight = 'balanced')

param_grid = {
    'n_estimators' : range(2,40,2),
    'max_depth' : range(2,10),
}

gs = GridSearchCV(model, param_grid = param_grid, scoring = 'accuracy')

gs.fit(X_train, y_train)

gs.best_estimator_

best_dt_estimates = gs.best_estimator_

best_dt_estimates.fit(X_train, y_train)

y_train_predicted = best_dt_estimates.predict(X_train)
y_test_predicted = best_dt_estimates.predict(X_test)

print(classification_report(y_test, y_test_predicted))

joblib.dump(best_dt_estimates,'final_model.pkl')

joblib.dump(list(X.columns),'column_names.pkl')