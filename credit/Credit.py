import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
data = pd.read_csv('data.csv')
print(data.dtypes)

data = pd.DataFrame(data)
# Preprocessing
preprocessing = 3
if(preprocessing==1):
    data = data.dropna()
elif(preprocessing==2):
    data = data.fillna(0)
elif(preprocessing==3):
    imp = SimpleImputer(strategy='mean')
    data = imp.fit_transform(data)



y = data['SeriousDlqin2yrs']
data = data.drop(columns=['Id', 'SeriousDlqin2yrs'])
X = data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(C=5.0)
lr.fit(X_train, y_train)

print("Training Score of Logistic Regression: ", accuracy_score(y_val, lr.predict(X_val)))
