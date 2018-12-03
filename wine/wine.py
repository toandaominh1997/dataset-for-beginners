import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
data = pd.read_csv('data.csv')
data = pd.DataFrame(data)
# Check is NULL
data.isna().sum()


y = data.iloc[:, 0]
data = data.drop(columns=[data.columns[0]], axis=1)
print(y.shape)
# # Check type
data.dtypes
for col in data.columns:
    if(data[col].dtypes=='object'):
        data = pd.concat([data.drop(columns=[col], axis=1), pd.get_dummies(data[col], prefix=col)], axis=1)
print(data.shape)
X = data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(C=10.0)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(max_depth=None)
rf.fit(X_train, y_train)

svc = SVC(C=5.0)
svc.fit(X_train, y_train)
print("Validation Score with Logistic Regression: ", accuracy_score(y_val, lr.predict(X_val)))

print("Validation Score with Random Forest Classifier: ", accuracy_score(y_val, rf.predict(X_val)))

print("Validation Score with Random Support Vector Machine: ", accuracy_score(y_val, svc.predict(X_val)))