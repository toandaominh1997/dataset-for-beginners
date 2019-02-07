import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Scale datasets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

# from sklearn.preprocessing import Normalizer

# normalizer = Normalizer()
# X = pd.DataFrame(normalizer.fit_transform(X))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
svc = SVC(C=20.0)
#lr.fit(X_train, y_train)
svc.fit(X_train, y_train)
lr.fit(X_train, y_train)
print("Accuracy: {:.2}, {:.2}".format(accuracy_score(y_train, svc.predict(X_train)), accuracy_score(y_val, svc.predict(X_val))))

h = .02

x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
colors = [color_map[i] for i in y]
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors)
plt.show()

