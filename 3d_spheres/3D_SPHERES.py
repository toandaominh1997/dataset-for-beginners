import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

print(data.shape)
print("Number is NULL: ", data.isnull().sum())

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

svc = SVC(C=5.0)
svc.fit(X_train, y_train)

def Plot(x1, x2, y):
    color_map = {1:(1, 0, .0), 2:(0, .3, .9)}
    color = [color_map[i] for i in y]
    plt.scatter(x1, x2, c=color)
    plt.show()
def Plott(x1, x2, y):
    x1_min, x1_max= x1.min()-1, x1.max()+1
    x2_min, x2_max = x2.min()-1, x2.max()+1
    xx, yy = np.meshgrid(np.arange(x1_min, x2_max, .02), np.arange(x2_min, x2_max, .02))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    color_map = {1:(1, 0, .0), 2:(0, .3, .9)}
    color = [color_map[i] for i in y]
    plt.scatter(x1, x2, c=color)
    plt.title("toan")
    plt.show()
Plot(X.iloc[:, 0], X.iloc[:, 1], y)
print("Training accuracy: ", accuracy_score(y_val, svc.predict(X_val)))

