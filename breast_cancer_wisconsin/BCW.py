import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

# print(df.isnull().sum())
# print(df.shape)
#print(df.nunique())
df.loc[df['diagnosis']=='M', 'diagnosis']=0
df.loc[df['diagnosis']=='B', 'diagnosis']=1
y = df['diagnosis']
df = df.drop(columns=['diagnosis', 'id'])
X = df

# for col in df.columns:
#     if(df[col].dtypes=='object'):
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(C=5.0)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(max_depth=10.0)
rf.fit(X_train, y_train)
# svc = SVC(C=1.0, kernel='linear')
# svc.fit(X_train, y_train)

# Accuracy Score
print("Training score of Logistic Regression: ", accuracy_score(y_val, lr.predict(X_val)))

print("Training score of Random forest: ", accuracy_score(y_val, rf.predict(X_val)))