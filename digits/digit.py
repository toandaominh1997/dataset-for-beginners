import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = pd.DataFrame(train)
test = pd.DataFrame(test)
for col in train.columns:
    if(train[col].nunique()==1):
        train=train.drop(columns=[col])
        

for col in test.columns:
    if(test[col].nunique()==1):
        test=test.drop(columns=[col])
X = train.iloc[:, :-1]
y = train.iloc[:, -1:]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(C=10.0)
lr.fit(X_train, y_train)

svc = SVC(C=5.0)
svc.fit(X_train, y_train)

rf = RandomForestClassifier(max_depth=None)
rf.fit(X_train, y_train)

print("Training score of LogisticRegression: ", accuracy_score(y_val, lr.predict(X_val)))
print("Training score of SVM: ", accuracy_score(y_val, svc.predict(X_val)))

print("Training score of Random Forest Classifier: ", accuracy_score(y_val, rf.predict(X_val)))
