import mplcursors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from pyod.models.knn import KNN
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/3D_spatial_network.csv").sample(n=500).reset_index(drop=True)
x = df.drop('alt', axis=1)
y = df.alt
min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x)
grid = np.linspace(0, 1, 80)
x_, y_ = np.meshgrid(grid, grid)
x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T

clf = KNN(n_neighbors=10)
clf.fit(x_train_scaled)
distance, neighbours = clf.get_neighbours(x_test_scaled)
y_test_scores = clf.decision_function(x_test_scaled)

f3_liberal = []
f3_conservative = []
uncertainty = []
outlierness = []

for idx, val in enumerate(x_test_scaled):
    neighborhood = [y[i] for i in neighbours[idx]]
    avg = np.ones(10) * np.mean(neighborhood)
    f1 = 10 * mean_squared_error(neighborhood, avg)
    f2 = y_test_scores[idx]
    # f3_conservative.append(f1 + f2 - (f1 * f2))
    # f3_liberal.append(f1 * f2)
    uncertainty.append(f1)
    outlierness.append(f2)

norm = np.linalg.norm(uncertainty)
uncertainty = uncertainty / norm
norm = np.linalg.norm(outlierness)
outlierness = outlierness / norm
f1 = uncertainty
f2 = outlierness

# f1 = stats.zscore(f1)
# f2 = stats.zscore(f2)
# print(f1)
# print(f2)

for i in range(len(uncertainty)):
    f3_conservative.append(f1[i] + f2[i] - (f1[i] * f2[i]))
    f3_liberal.append(f1[i] * f2[i])

# Plot
minima = np.min(f3_liberal)
maxima = np.max(f3_liberal)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)

color = []
for v in f3_liberal:
    color.append(mapper.to_rgba(v))
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
scatter = axs[0].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                         color=[item for item in color], s=2)
axs[0].set_aspect('equal', adjustable='box')
axs[0].title.set_text("Liberal")
cursor = mplcursors.cursor(scatter, hover=True)


@cursor.connect("add")
def on_add(sel):
    sel.annotation.set(text=f3_liberal[sel.target.index])


minima = np.min(f3_conservative)
maxima = np.max(f3_conservative)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
color = []
for v in f3_conservative:
    color.append(mapper.to_rgba(v))
scatter = axs[1].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                         color=[item for item in color], s=2)
axs[1].set_aspect('equal', adjustable='box')
axs[1].title.set_text("Conservative")
cursor = mplcursors.cursor(scatter, hover=True)


@cursor.connect("add")
def on_add(sel):
    sel.annotation.set(text=f3_conservative[sel.target.index])


plt.show()

minima = np.min(outlierness)
maxima = np.max(outlierness)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)

color = []
for v in outlierness:
    color.append(mapper.to_rgba(v))
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
scatter = axs[0].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                         color=[item for item in color], s=2)
axs[0].set_aspect('equal', adjustable='box')
axs[0].title.set_text("Only Outlierness")
cursor = mplcursors.cursor(scatter, hover=True)


@cursor.connect("add")
def on_add(sel):
    sel.annotation.set(text=outlierness[sel.target.index])


minima = np.min(uncertainty)
maxima = np.max(uncertainty)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
color = []
for v in uncertainty:
    color.append(mapper.to_rgba(v))
scatter = axs[1].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                         color=[item for item in color], s=2)
axs[1].set_aspect('equal', adjustable='box')
axs[1].title.set_text("Only Uncertainty")
cursor = mplcursors.cursor(scatter, hover=True)


@cursor.connect("add")
def on_add(sel):
    sel.annotation.set(text=uncertainty[sel.target.index])


plt.show()

plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], s=10)
plt.show()
