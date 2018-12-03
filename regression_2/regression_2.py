import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score
data = pd.read_csv('data.csv')
data = pd.DataFrame(data)
# Check is NULL
data = data.fillna(0)
data.isnull().sum()

y = data['y']
data = data.drop(columns=['y'], axis=1)
print(y.shape)
# # Check type
data.dtypes
for col in data.columns:
    if(data[col].dtypes=='object'):
        data = pd.concat([data.drop(columns=[col], axis=1), pd.get_dummies(data[col], prefix=col)], axis=1)
print(data.shape)
X = data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("MSE of Validation: ", mean_squared_error(y_val, lr.predict(X_val)))