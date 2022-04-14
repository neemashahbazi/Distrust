import math

import numpy as np
import pandas as pd
from numpy import searchsorted
from pyod.models.knn import KNN
from scipy.stats import norm
from sklearn import preprocessing
from matplotlib import pyplot as plt, cm
import mplcursors
from matplotlib.colors import Normalize


def in_radius(c_x, c_y, r, x, y):
    return math.hypot(c_x - x, c_y - y) <= r


mean = [0, 0]
cov = [[4, 0], [0, 1]]
label = []

x, y = np.random.multivariate_normal(mean, cov, 1000).T
for i in range(len(x)):
    if in_radius(-1.5, -1.5, 2.5, x[i], y[i]):
        label.append(0)
    else:
        label.append(1)

d = {'X': x, 'Y': y, 'Label': label}
df = pd.DataFrame(data=d)
x = df.drop('Label', axis=1)
y = df.Label
min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x)

grid = np.linspace(0, 1, 80)
x_, y_ = np.meshgrid(grid, grid)
x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T

clf = KNN(n_neighbors=10)
clf.fit(x_train_scaled)
distance, neighbours = clf.get_neighbours(x_test_scaled)
y_train_scores = clf.decision_scores_
y_test_scores = clf.decision_function(x_test_scaled)
# ---------------------------------------------------
mean = 0.9
sd = 0.1
y_train_scores.sort()
outlier_mean_index = int(mean * len(y_train_scores))
outlier_mean = y_train_scores[outlier_mean_index]
prob = []
uncertain=[]
for idx, val in enumerate(y_test_scores):
    query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
    percentile = (mean * query_index) / outlier_mean_index
    z_score = (percentile - mean) / sd
    prob.append(norm.cdf(z_score))
    uncertain.append(val)
    # print("Percentile: ", percentile)
    # print("Z-Score: ", z_score)
    # print("CDF: ", norm.cdf(z_score))
    # print("---------------------------")

# Plot
minima = np.min(prob)
maxima = np.max(prob)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)

color = []
for v in prob:
    color.append(mapper.to_rgba(v))
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
scatter = axs[0].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                         color=[item for item in color], s=2)
axs[0].set_aspect('equal', adjustable='box')
axs[0].title.set_text("Probability Z_score")

cursor = mplcursors.cursor(scatter, hover=True)


@cursor.connect("add")
def on_add(sel):
    sel.annotation.set(text=prob[sel.target.index])


minima = np.min(prob)
maxima = np.max(prob)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)

color = []
for v in uncertain:
    color.append(mapper.to_rgba(v))
scatter = axs[1].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                         color=[item for item in color], s=2)
axs[1].set_aspect('equal', adjustable='box')
axs[1].title.set_text("Probability_pure")

cursor = mplcursors.cursor(scatter, hover=True)


@cursor.connect("add")
def on_add(sel):
    sel.annotation.set(text=prob[sel.target.index])

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
colors = np.where(y == 0, 'r', 'b')
axs[1].scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], s=10, c=colors)
plt.show()
