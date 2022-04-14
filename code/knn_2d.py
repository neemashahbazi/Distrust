import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import searchsorted
from pyod.models.knn import KNN
from scipy.stats import entropy
from scipy.stats import norm


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


num_of_points_list = [50, 100, 200, 300, 400, 500, 1000, 5000, 10000, 50000, 100000]
num_of_neighbors_list = [1, 2, 5, 10, 20, 30, 40, 50]
contamination_rate_list = [0.05, 0.1, 0.15, 0.2, 0.25]
default_num_of_points = 1000
default_num_of_neighbors = 10
default_contamination_rate = 0.1
duration_list_get_neighbours = []
duration_list_binary_search = []
duration_list_total = []
gaussian_mean = [0, 0]
gaussian_cov = [[6, 4], [3, 1]]

# for num_of_points in num_of_points_list:
num_of_points = default_num_of_points

# iteration_list_get_neighbours = []
# iteration_list_binary_search = []
# iteration_list_total = []

    # for iteration in range(30):
# folder = "data/synthetic/num_of_points" + "_" + str(num_of_points)+"/"+str(iteration)+".csv"
# df = pd.read_csv(folder)
# x = df.drop('Y', axis=1)
# y = df.Y
# grid = np.linspace(0, 1, 80)
# x_, y_ = np.meshgrid(grid, grid)
# x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T

# for num_of_neighbors in num_of_neighbors_list:
for contamination_rate in contamination_rate_list:

    num_of_neighbors = default_num_of_neighbors
    # contamination_rate = default_contamination_rate
    iteration_list_get_neighbours = []
    iteration_list_binary_search = []
    iteration_list_total = []

    for iteration in range(30):
        folder = "data/synthetic/num_of_points" + "_" + str(num_of_points)+"/"+str(iteration)+".csv"
        df = pd.read_csv(folder)
        x = df.drop('Y', axis=1)
        y = df.Y
        grid = np.linspace(0, 1, 80)
        x_, y_ = np.meshgrid(grid, grid)
        x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T

        clf = KNN(n_neighbors=num_of_neighbors)
        clf.fit(x)
        neighbours = []
        duration_list_get_neighbours_avg = []
        for test_item in x_test_scaled:
            start = time.time()
            distance, neighbour = clf.get_neighbours(test_item.reshape(1, -1))
            end = time.time()
            neighbours.append(neighbour)
            duration = end - start
            duration_list_get_neighbours_avg.append(duration)
        avg_get_neighbour = np.mean(duration_list_get_neighbours_avg)
        iteration_list_get_neighbours.append(avg_get_neighbour)
        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(x_test_scaled)
        f3_liberal = []
        f3_conservative = []
        mean = 1.0 - contamination_rate
        sd = 0.1
        y_train_scores.sort()
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        for idx, val in enumerate(y_test_scores):
            labels = [y[i] for i in neighbours[idx]]
            f1 = shannon_entropy(labels)
            start = time.time()
            query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
            end = time.time()
            duration = end - start
            duration_list_binary_search_avg.append(duration)
            percentile = (mean * query_index) / outlier_mean_index
            z_score = (percentile - mean) / sd
            f2 = norm.cdf(z_score)
            f3_conservative.append(f1 + f2 - (f1 * f2))
            f3_liberal.append(f1 * f2)
        avg_binary_search = np.mean(duration_list_binary_search_avg)
        iteration_list_binary_search.append(avg_binary_search)
        avg_total = avg_get_neighbour + avg_binary_search
        iteration_list_total.append(avg_total)
    duration_list_get_neighbours.append(np.mean(iteration_list_get_neighbours))
    duration_list_binary_search.append(np.mean(iteration_list_binary_search))
    duration_list_total.append(np.mean(iteration_list_total))

# ------------------------------------------------------------
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# axs.plot(num_of_neighbors_list, duration_list_get_neighbours)
# axs.title.set_text("Temporal Performance: get_neighbours()")
# axs.set_ylabel('Duration')
# axs.set_xlabel('Number of Neighbours')
# plt.savefig("results/time_var_num_of_neighbours_get_neighbours.png")
#
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# axs.plot(num_of_neighbors_list, duration_list_binary_search)
# axs.title.set_text("Temporal Performance: binary_search()")
# axs.set_ylabel('Duration')
# axs.set_xlabel('Number of Neighbours')
# plt.savefig("results/time_var_num_of_neighbours_binary_search.png")
#
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# axs.plot(num_of_neighbors_list, duration_list_total)
# axs.title.set_text("Temporal Performance: Total")
# axs.set_ylabel('Duration')
# axs.set_xlabel('Number of Neighbours')
# plt.savefig("results/time_var_num_of_neighbours_total.png")

# ------------------------------------------------------------
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# axs.plot(num_of_points_list, duration_list_get_neighbours)
# axs.title.set_text("Temporal Performance: get_neighbours()")
# axs.set_ylabel('Duration')
# axs.set_xlabel('Number of Points')
# axs.set_xscale('log')
# plt.savefig("results/time_var_num_of_points_get_neighbours.png")
#
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# axs.plot(num_of_points_list, duration_list_binary_search)
# axs.title.set_text("Temporal Performance: binary_search()")
# axs.set_ylabel('Duration')
# axs.set_xlabel('Number of Points')
# axs.set_xscale('log')
# plt.savefig("results/time_var_num_of_points_binary_search.png")
#
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# axs.plot(num_of_points_list, duration_list_total)
# axs.title.set_text("Temporal Performance: Total")
# axs.set_ylabel('Duration')
# axs.set_xlabel('Number of Points')
# axs.set_xscale('log')
# plt.savefig("results/time_var_num_of_points_total.png")

# ------------------------------------------------------------
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(contamination_rate_list, duration_list_get_neighbours)
axs.title.set_text("Temporal Performance: get_neighbours()")
axs.set_ylabel('Duration')
axs.set_xlabel('Contamination Rate')
plt.savefig("results/time_var_contamination_rate_get_neighbours.png")

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(contamination_rate_list, duration_list_binary_search)
axs.title.set_text("Temporal Performance: binary_search()")
axs.set_ylabel('Duration')
axs.set_xlabel('Contamination Rate')
plt.savefig("results/time_var_contamination_rate_binary_search.png")

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(contamination_rate_list, duration_list_total)
axs.title.set_text("Temporal Performance: Total")
axs.set_ylabel('Duration')
axs.set_xlabel('Contamination Rate')
plt.savefig("results/time_var_contamination_rate_total.png")
