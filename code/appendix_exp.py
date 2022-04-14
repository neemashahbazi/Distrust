import matplotlib
import mplcursors
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import searchsorted
from pyod.models.knn import KNN
from scipy.stats import entropy
from scipy.stats import norm
from sklearn import preprocessing


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

font = {
    'size': 25}

matplotlib.rc('font', **font)
num_of_neighbors = 10
contamination_rate = 0.1

centers = [(0, 3.5), (-2, 0), (2, 0)]
covs = [np.eye(2), np.eye(2) * 2, np.diag([5, 1])]
x_min, x_max, y_min, y_max, step = -6, 8, -6, 8, 0.1
n_samples = 500
n_classes = 3
np.random.seed(42)
X = np.vstack(
    [
        np.random.multivariate_normal(center, cov, n_samples)
        for center, cov in zip(centers, covs)
    ]
)
y = np.hstack([np.full(n_samples, i) for i in range(n_classes)])
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X)
y_train = y

grid = np.linspace(0, 1, 80)
x_, y_ = np.meshgrid(grid, grid)
X_cal = np.vstack([x_.ravel(), y_.ravel()]).T

clf = KNN(n_neighbors=num_of_neighbors)
clf.fit(X_train)
distance, neighbours = clf.get_neighbours(X_cal)
y_train_scores = clf.decision_scores_
y_test_scores = clf.decision_function(X_cal)
f3_liberal = []
f3_conservative = []

mean = 1.0 - contamination_rate
sd = 0.1
y_train_scores.sort()
outlier_mean_index = int(mean * len(y_train_scores))
outlier_mean = y_train_scores[outlier_mean_index]

for idx, val in enumerate(y_test_scores):
    labels = [y_train[i] for i in neighbours[idx]]
    f1 = shannon_entropy(labels)
    query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
    percentile = (mean * query_index) / outlier_mean_index
    z_score = (percentile - mean) / sd
    f2 = norm.cdf(z_score)
    f3_conservative.append(f1 + f2 - (f1 * f2))
    f3_liberal.append(f1 * f2)

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
minima = np.min(f3_conservative)
maxima = np.max(f3_conservative)
normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
color = []
for v in f3_conservative:
    color.append(mapper.to_rgba(v))
scatter = ax.scatter([item[0] for item in X_cal], [item[1] for item in X_cal],
                     color=[item for item in color], s=7)
ax.set_aspect('equal', adjustable='box')
ax.title.set_text("WDT")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
cursor = mplcursors.cursor(scatter, hover=True)
plt.savefig("results/extension/wdt.png")

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
minima = np.min(f3_liberal)
maxima = np.max(f3_liberal)
normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
color = []
for v in f3_liberal:
    color.append(mapper.to_rgba(v))
scatter = ax.scatter([item[0] for item in X_cal], [item[1] for item in X_cal],
                     color=[item for item in color], s=7)
ax.set_aspect('equal', adjustable='box')
ax.title.set_text("SDT")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
cursor = mplcursors.cursor(scatter, hover=True)
plt.savefig("results/extension/sdt.png")
