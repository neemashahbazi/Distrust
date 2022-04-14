import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def best_lr(x_train_scaled, y):
    lr = LogisticRegression()
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'max_iter': list(range(100, 800, 100)),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    lr_search = GridSearchCV(lr, param_grid=lr_param_grid, refit=True, verbose=3, cv=5)
    lr_search.fit(x_train_scaled, y)
    print('Mean Accuracy: %.3f' % lr_search.best_score_)
    print('Config: %s' % lr_search.best_params_)
    return lr_search.best_params_


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

    x_min, x_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    axs.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap="RdYlBu", edgecolor="white")
    axs.scatter(-7, -7, alpha=1.0, s=90, c='k', edgecolor="white")
    axs.scatter(2, 0, alpha=1.0, s=90, c='k', edgecolor="white")
    axs.scatter(8, 6, alpha=1.0, s=90, c='k', edgecolor="white")
    axs.scatter(-2.5, -1, alpha=1.0, s=90, c='k', edgecolor="white")
    plt.annotate('$q^1$', (-6.7, -6.7), fontsize=15)
    plt.annotate('$q^2$', (2.3, 0.3), fontsize=15)
    plt.annotate('$q^3$', (8.3, 6.3), fontsize=15)
    plt.annotate('$q^4$', (-2.2, -0.7), fontsize=15)

    axs.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    axs.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])

    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis('equal')
    plt.savefig("results/example_2.png")
    # plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    axs.scatter(-7, -7, alpha=1.0, s=90, c='k', edgecolor="white")
    axs.scatter(2, 0, alpha=1.0, s=90, c='k', edgecolor="white")
    axs.scatter(8, 6, alpha=1.0, s=90, c='k', edgecolor="white")
    axs.scatter(-2.5, -1, alpha=1.0, s=90, c='k', edgecolor="white")
    plt.annotate('$q^1$', (-6.7, -6.7), fontsize=15)
    plt.annotate('$q^2$', (2.3, 0.3), fontsize=15)
    plt.annotate('$q^3$', (8.3, 6.3), fontsize=15)
    plt.annotate('$q^4$', (-2.2, -0.7), fontsize=15)

    axs.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    axs.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])

    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis('equal')
    circle = plt.Circle((-4.5, -3.5), 3.5, color='r', alpha=0.6)
    axs.add_patch(circle)
    plt.savefig("results/example_3.png")
    # plt.show()


def in_radius(c_x, c_y, r, x, y):
    return math.hypot(c_x - x, c_y - y) <= r


mean = [0, 0]
cov = [[6, 4], [3, 1]]
label = []

x, y = np.random.multivariate_normal(mean, cov, 300).T
for i in range(len(x)):
    if in_radius(-4.5, -3.5, 3.5, x[i], y[i]):
        label.append(0)
    else:
        label.append(1)

d = {'X': x, 'Y': y, 'Label': label}
df = pd.DataFrame(data=d)
x = df.drop('Label', axis=1)
y = df.Label
x = x.to_numpy()

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
colors = np.where(y == 0, 'r', 'b')
axs.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8, cmap="RdYlBu", edgecolor="white")
plt.axis('equal')
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

axs.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
axs.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])

# plt.savefig("results/example_1.png")
plt.show()

params = best_lr(x, y)
plot_decision_boundaries(x, y, LogisticRegression, **params)
