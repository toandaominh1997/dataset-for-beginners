import numpy as np 
import pandas as pd 
import os 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('data.csv')
# print("Number data is null: ", data.isnull().sum())
# print("Description: ", data.describe())
# print("Types: ", data.dtypes)
# print("Nunique: ", data.nunique())
df = pd.DataFrame(data)
df.dropna() 
df.loc[df[' income']==' <=50K', ' income']=0
df.loc[df[' income']==' >50K', ' income'] = 1
y = df[' income']
df = df.drop(columns=[' income'])
print("Before Shape: ", df.shape)
for col in df.columns:
    if(df[col].dtypes=='object'):
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df = df.drop(columns=[col])
# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=42)

#Model using SVM with C = 5.0 and kernel = 'rbf' 
svc = SVC(C=5.0)
#svc.fit(X_train, y_train)

# Model: Logistic Regression
lr = LogisticRegression(C=20.0)
#lr.fit(X_train, y_train)

# Model: RandomForest 
rf = RandomForestClassifier(max_depth=10.0)
rf.fit(X_train, y_train)
# Training accuracy:  0.8235836020267158
# print("Training accuracy of SVM ", accuracy_score(y_val, svc.predict(X_val)))

# print("Training accuracy of LR: ", accuracy_score(y_val, lr.predict(X_val)))
print("Training accuracy of Random Forest: ", accuracy_score(y_val, rf.predict(X_val)))

# Best Accuracy is Random Forest: 0.8527560264087211 with max_depth = 10.0