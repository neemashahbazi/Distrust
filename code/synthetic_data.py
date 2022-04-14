import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from pyod.models.knn import KNN
from pyod.utils.example import visualize
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def plot_decision_boundaries(X, y, model_class, **model_params):
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature-1", fontsize=15)
    plt.ylabel("Feature-2", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt


def in_radius(c_x, c_y, r, x, y):
    return math.hypot(c_x - x, c_y - y) <= r


def best_gmm(x_train_scaled):
    n_components = np.arange(1, 40)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(x_train_scaled)
              for n in n_components]

    plt.plot(n_components, [m.bic(x_train_scaled) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(x_train_scaled) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()


def best_lr(x_train_scaled):
    LR = LogisticRegression()
    LRparam_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'max_iter': list(range(100, 800, 100)),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit=True, verbose=3, cv=5)

    # fitting the model for grid search
    LR_search.fit(x_train_scaled, y)
    # summarize
    print('Mean Accuracy: %.3f' % LR_search.best_score_)
    print('Config: %s' % LR_search.best_params_)


mean = [0, 0]
cov = [[4, 0], [0, 1]]

# x_blue = []
# y_blue = []
# x_red = []
# y_red = []
label = []

x, y = np.random.multivariate_normal(mean, cov, 500).T
for i in range(len(x)):
    if in_radius(-1.5, -1.5, 2.5, x[i], y[i]):
        # x_red.append(x[i])
        # y_red.append((y[i]))
        label.append(0)
    else:
        # x_blue.append(x[i])
        # y_blue.append(y[i])
        label.append(1)

# plt.plot(x_blue, y_blue, 'bx')
# plt.plot(x_red, y_red, 'r+')
# plt.axis('equal')
# plt.show()
# ----------------------------------------------------------------------
d = {'X': x, 'Y': y, 'Label': label}
df = pd.DataFrame(data=d)
x = df.drop('Label', axis=1)
y = df.Label
min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x)

# ----------------------------------------------------------------------
clf = KNN()
clf.fit(x_train_scaled)

y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_
print(y_train_scores)
visualize("KNN", x_train_scaled, y, x_train_scaled, y, y_train_pred,
          y_train_pred, show_figure=True, save_figure=False)
# ----------------------------------------------------------------------

# best_lr(x_train_scaled)
lr = LogisticRegression(C=100, max_iter=100, penalty='l1', solver='liblinear')
lr.fit(x_train_scaled, y)

# best_gmm(x_train_scaled)
gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
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

# grid = np.linspace(0, 1, 500)
# x_, y_ = np.meshgrid(grid, grid)
# X_ = np.vstack([x_.ravel(), y_.ravel()]).T
# logprob = gmm.score_samples(X_)
# print(np.exp(logprob).sum() * (grid[1] - grid[0]) ** 2)

f1 = min_max_scaler.fit_transform(f1)
f2 = min_max_scaler.fit_transform(f2)
# f2 = min_max_scaler.fit_transform(np.array(f2).reshape(-1, 1))
for i in range(len(f2)):
    f3.append(f1[i] * f2[i])
    # print("(" + str(x_test[i]) + " , " + str(y_test[i]) + ")")
    # print("LR Prob:" + str(f1[i]))
    # print("GMM Score:" + str(f2[i]))
    # print("_____________________________")

# f3 = f2

# f3 = min_max_scaler.fit_transform(f3)
# for i in range(len(f3)):
#     print("(" + str(x_test[i]) + " , " + str(y_test[i]) + ")")
#     print("Trust:" + str(f3[i]))
#     print("_____________________________")

minima = np.min(f3)
maxima = np.max(f3)
print(minima)
print(maxima)
norm = Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn)

color = []
for v in f3:
    color.append(mapper.to_rgba(v))

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

for idx in range(len(x_test)):
    axs[0].scatter(x_test[idx], y_test[idx], color=color[idx], s=2)

colors = np.where(y == 0, 'r', 'b')
axs[1].scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], marker='x', c=colors)
axs[0].set_aspect('equal', adjustable='box')
axs[1].set_aspect('equal', adjustable='box')
plt.show()

plot_decision_boundaries(x_train_scaled, y, LogisticRegression)
