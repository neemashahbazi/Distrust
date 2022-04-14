import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

df = pd.read_csv("data/iris.csv")
x = df.drop('X_3', axis=1)
x = x.drop('X_4', axis=1)
x = x.drop('Y', axis=1)
y = df.Y
min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x)

lr = LogisticRegression()
lr.fit(x_train_scaled, y)

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(x_train_scaled)

x_test = []
y_test = []
f1 = []
f2 = []
f3 = []
for i in np.arange(0, 1, 0.02):
    for j in np.arange(0, 1, 0.02):
        f1.append(abs(lr.predict_proba([[i, j]])[:, 1] - 0.5))
        f2.append(np.exp(gmm.score_samples([[i, j]])))
        x_test.append(i)
        y_test.append(j)
f1 = min_max_scaler.fit_transform(f1)
f2 = min_max_scaler.fit_transform(f2)
for i in range(len(f2)):
    f3.append(f1[i] + f2[i])
    print("(" + str(x_test[i]) + " , " + str(y_test[i]) + ")")
    print("LR Prob:" + str(f1[i]))
    print("GMM Score:" + str(f2[i]))
    print("_____________________________")

f3 = f1
minima = np.min(f3)
maxima = np.max(f3)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn)

color = []
for v in f3:
    color.append(mapper.to_rgba(v))

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

for idx in range(len(x_test)):
    axs[0].scatter(x_test[idx], y_test[idx], color=color[idx], s=2)

colors = np.where(y == 'Iris-setosa', 'r', 'b')
axs[1].scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], marker='o', c=colors)
axs[0].set_aspect('equal', adjustable='box')
axs[1].set_aspect('equal', adjustable='box')
plt.show()
