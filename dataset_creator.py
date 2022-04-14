import math
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing


def in_radius(c_x, c_y, r, x, y):
    return math.hypot(c_x - x, c_y - y) <= r


def in_shape(x, y, dir):
    image = cv2.imread(dir)
    if (image[int(x * 1000) - 1][int(y * 1000) - 1] == [255, 255, 255]).all():
        return True
    else:
        return False


gaussian_mean = [0, 0]
gaussian_cov = [[6, 4], [3, 1]]
num_of_points = 1000
# x_1, x_2 = np.random.multivariate_normal(gaussian_mean, gaussian_cov, num_of_points).T
# d = {'X_1': x_1, 'X_2': x_2}
# x = pd.DataFrame(data=d)
# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)
# label = []
# for i in range(len(x)):
#     if in_shape(x[i][0], x[i][1], 'data/shapes/11.png'):
#         label.append(0)
#     else:
#         label.append(1)
#
# y = np.array(label)
#
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# colors = np.where(y == 0, 'r', 'b')
# axs.scatter(x[:, 0], x[:, 1], s=10, c=colors)
# plt.show()
for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    folder_train = "data/synthetic/train/"
    folder_test = "data/synthetic/test/"

    if not os.path.exists(folder_train):
        os.makedirs(folder_train)
    if not os.path.exists(folder_test):
        os.makedirs(folder_test)
    for iteration in range(5):
        label = []
        x_1, x_2 = np.random.multivariate_normal(gaussian_mean, gaussian_cov, num_of_points).T
        d = {'X_1': x_1, 'X_2': x_2}
        x_train = pd.DataFrame(data=d)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train)
        for i in range(len(x_train)):
            if in_shape(x_train[i][0], x_train[i][1], 'data/shapes/' + str(index) + '.png'):
                label.append(0)
            else:
                label.append(1)

        y_train = np.array(label)
        d = {'X_1': x_train[:, 0], 'X_2': x_train[:, 1], 'Y': y_train}
        train_data = pd.DataFrame(data=d)
        train_data.to_csv(folder_train + str((index - 1) * 5 + iteration + 1) + '.csv', index=False)

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        colors = np.where(y_train == 0, 'r', 'b')
        axs.scatter(x_train[:, 0], x_train[:, 1], s=10, c=colors)
        plt.savefig(folder_train + str((index - 1) * 5 + iteration + 1) + '.png')

        grid = np.linspace(0, 1, 80)
        x_, y_ = np.meshgrid(grid, grid)
        x_test = np.vstack([x_.ravel(), y_.ravel()]).T

        label = []
        for i in range(len(x_test)):
            if in_shape(x_test[i][0], x_test[i][1], 'data/shapes/' + str(index) + '.png'):
                label.append(0)
            else:
                label.append(1)

        y_test = np.array(label)
        d = {'X_1': x_test[:, 0], 'X_2': x_test[:, 1], 'Y': y_test}
        test_data = pd.DataFrame(data=d)
        test_data.to_csv(folder_test + str((index - 1) * 5 + iteration + 1) + '.csv', index=False)

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        colors = np.where(y_test == 0, 'r', 'b')
        axs.scatter(x_test[:, 0], x_test[:, 1], s=10, c=colors)
        plt.savefig(folder_test + str((index - 1) * 5 + iteration + 1) + '.png')

# #///////////////
# import math
# import os
# import time
#
# import mplcursors
# from scipy.stats import norm
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt, cm
# from matplotlib.colors import Normalize
# from numpy import searchsorted
# from pyod.models.knn import KNN
# from scipy.stats import entropy
# from sklearn import preprocessing
#
#
# def in_radius(c_x, c_y, r, x, y):
#     return math.hypot(c_x - x, c_y - y) <= r
#
#
# def shannon_entropy(labels, base=None):
#     value, counts = np.unique(labels, return_counts=True)
#     return entropy(counts, base=base)
#
#
# num_of_points_list = [50, 100, 200, 300, 400, 500, 1000, 5000, 10000, 50000, 100000]
# num_of_neighbors_list = [1, 2, 5, 10, 20, 30, 40, 50]
# contamination_rate_list = [0.05, 0.1, 0.15, 0.2, 0.25]
# default_num_of_points = 1000
# default_num_of_neighbors = 10
# default_contamination_rate = 0.1
# duration_list_get_neighbours = []
# duration_list_binary_search = []
# duration_list_total = []
#
# gaussian_mean = [0, 0]
# gaussian_cov = [[6, 4], [3, 1]]
#
# for num_of_points in num_of_points_list:
#     folder = "results/var_num_of_points" + "_" + str(num_of_points)
#     num_of_points = default_num_of_points
#
#     iteration_list_get_neighbours = []
#     iteration_list_binary_search = []
#     iteration_list_total = []
#
#     for iteration in range(30):
#         folder = "data/synthetic/num_of_points" + "_" + str(num_of_points)+str(iteration)+".csv"
#         df = pd.read_csv(folder)
#         x = df.drop('alt', axis=1)
#         y = df.alt
#         # label = []
#         # x_1, x_2 = np.random.multivariate_normal(gaussian_mean, gaussian_cov, num_of_points).T
#         # d = {'X_1': x_1, 'X_2': x_2}
#         # x = pd.DataFrame(data=d)
#         # min_max_scaler = preprocessing.MinMaxScaler()
#         # x_train_scaled = min_max_scaler.fit_transform(x)
#         #
#         # for i in range(len(x_train_scaled)):
#         #     if in_radius(0.30, 0.30, 0.22, x_train_scaled[i][0], x_train_scaled[i][1]):
#         #         label.append(0)
#         #     else:
#         #         label.append(1)
#         #
#         # x = x_train_scaled
#         # y = np.array(label)
#
#         grid = np.linspace(0, 1, 80)
#         x_, y_ = np.meshgrid(grid, grid)
#         x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T
#
#         # for num_of_neighbors in num_of_neighbors_list:
#         # for contamination_rate in contamination_rate_list:
#         #     folder = "results/var_num_of_neighbors" + "_" + str(num_of_neighbors)
#         #     folder = "results/var_contamination_rate" + "_" + str(contamination_rate)
#
#         num_of_neighbors = default_num_of_neighbors
#         contamination_rate = default_contamination_rate
#
#         clf = KNN(n_neighbors=num_of_neighbors)
#         clf.fit(x)
#         neighbours = []
#         duration_list_get_neighbours_avg = []
#         for test_item in x_test_scaled:
#             start = time.time()  # start timer
#             distance, neighbour = clf.get_neighbours(test_item.reshape(1, -1))
#             end = time.time()  # stop timer
#             neighbours.append(neighbour)
#             duration = end - start
#             duration_list_get_neighbours_avg.append(duration)
#         # if not os.path.exists(folder):
#         #     os.makedirs(folder)
#         # f = open(folder + "/" + "time_get_neighbours.txt", "w")
#         # f.write(str(avg_get_neighbour))
#         # f.close()
#
#         avg_get_neighbour = np.mean(duration_list_get_neighbours_avg)
#         iteration_list_get_neighbours.append(avg_get_neighbour)
#
#         y_train_scores = clf.decision_scores_
#         y_test_scores = clf.decision_function(x_test_scaled)
#
#         f3_liberal = []
#         f3_conservative = []
#         uncertainty = []
#         outlierness = []
#
#         mean = 1.0 - contamination_rate
#         sd = 0.1
#         y_train_scores.sort()
#         outlier_mean_index = int(mean * len(y_train_scores))
#         outlier_mean = y_train_scores[outlier_mean_index]
#
#         duration_list_binary_search_avg = []
#
#         for idx, val in enumerate(y_test_scores):
#             labels = [y[i] for i in neighbours[idx]]
#             f1 = shannon_entropy(labels)
#             start = time.time()  # start timer
#             query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
#             end = time.time()  # stop timer
#             duration = end - start
#             duration_list_binary_search_avg.append(duration)
#             percentile = (mean * query_index) / outlier_mean_index
#             z_score = (percentile - mean) / sd
#             f2 = norm.cdf(z_score)
#             f3_conservative.append(f1 + f2 - (f1 * f2))
#             f3_liberal.append(f1 * f2)
#             uncertainty.append(f1)
#             outlierness.append(f2)
#         avg_binary_search = np.mean(duration_list_binary_search_avg)
#         iteration_list_binary_search.append(avg_binary_search)
#         # f = open(folder + "/" + "time_binary_search.txt", "w")
#         # f.write(str(avg_binary_search))
#         # f.close()
#         #
#         avg_total = avg_get_neighbour + avg_binary_search
#         iteration_list_total.append(avg_total)
#         # f = open(folder + "/" + "time_total.txt", "w")
#         # f.write(str(avg_total))
#         # f.close()
#         #
#         # # Plot
#         # minima = np.min(f3_liberal)
#         # maxima = np.max(f3_liberal)
#         # normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
#         # mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
#         #
#         # color = []
#         # for v in f3_liberal:
#         #     color.append(mapper.to_rgba(v))
#         # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#         # scatter = axs[0].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#         #                          color=[item for item in color], s=2)
#         # axs[0].set_aspect('equal', adjustable='box')
#         # axs[0].title.set_text("Liberal")
#         # cursor = mplcursors.cursor(scatter, hover=True)
#         #
#         #
#         # @cursor.connect("add")
#         # def on_add(sel):
#         #     sel.annotation.set(text=f3_liberal[sel.target.index])
#         #
#         #
#         # minima = np.min(f3_conservative)
#         # maxima = np.max(f3_conservative)
#         # normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
#         # mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
#         # color = []
#         # for v in f3_conservative:
#         #     color.append(mapper.to_rgba(v))
#         # scatter = axs[1].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#         #                          color=[item for item in color], s=2)
#         # axs[1].set_aspect('equal', adjustable='box')
#         # axs[1].title.set_text("Conservative")
#         # cursor = mplcursors.cursor(scatter, hover=True)
#         #
#         #
#         # @cursor.connect("add")
#         # def on_add(sel):
#         #     sel.annotation.set(text=f3_conservative[sel.target.index])
#         #
#         #
#         # plt.savefig(folder + "/" + "1.png")
#         # # plt.show()
#         #
#         # minima = np.min(outlierness)
#         # maxima = np.max(outlierness)
#         # normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
#         # mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
#         #
#         # color = []
#         # for v in outlierness:
#         #     color.append(mapper.to_rgba(v))
#         # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#         # scatter = axs[0].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#         #                          color=[item for item in color], s=2)
#         # axs[0].set_aspect('equal', adjustable='box')
#         # axs[0].title.set_text("Outlierness (distance-only probabilty)")
#         # cursor = mplcursors.cursor(scatter, hover=True)
#         #
#         #
#         # @cursor.connect("add")
#         # def on_add(sel):
#         #     sel.annotation.set(text=outlierness[sel.target.index])
#         #
#         #
#         # minima = np.min(uncertainty)
#         # maxima = np.max(uncertainty)
#         # normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
#         # mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
#         # color = []
#         # for v in uncertainty:
#         #     color.append(mapper.to_rgba(v))
#         # scatter = axs[1].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#         #                          color=[item for item in color], s=2)
#         # axs[1].set_aspect('equal', adjustable='box')
#         # axs[1].title.set_text("Uncertainty (entropy)")
#         # cursor = mplcursors.cursor(scatter, hover=True)
#         #
#         #
#         # @cursor.connect("add")
#         # def on_add(sel):
#         #     sel.annotation.set(text=uncertainty[sel.target.index])
#         #
#         #
#         # plt.savefig(folder + "/" + "2.png")
#         # # plt.show()
#         #
#         # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
#         # colors = np.where(y == 0, 'r', 'b')
#         # axs.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], s=10, c=colors)
#         #
#         # plt.savefig(folder + "/" + "3.png")
#         # # plt.show()
#
#     duration_list_get_neighbours.append(np.mean(iteration_list_get_neighbours))
#     duration_list_binary_search.append(np.mean(iteration_list_binary_search))
#     duration_list_total.append(np.mean(iteration_list_total))
#
# # ------------------------------------------------------------
# # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# # axs.plot(num_of_neighbors_list, duration_list_get_neighbours)
# # axs.title.set_text("Temporal Performance: get_neighbours()")
# # axs.set_ylabel('Duration')
# # axs.set_xlabel('Number of Neighbours')
# # plt.savefig("results/time_var_num_of_neighbours_get_neighbours.png")
# #
# # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# # axs.plot(num_of_neighbors_list, duration_list_binary_search)
# # axs.title.set_text("Temporal Performance: binary_search()")
# # axs.set_ylabel('Duration')
# # axs.set_xlabel('Number of Neighbours')
# # plt.savefig("results/time_var_num_of_neighbours_binary_search.png")
# #
# # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# # axs.plot(num_of_neighbors_list, duration_list_total)
# # axs.title.set_text("Temporal Performance: Total")
# # axs.set_ylabel('Duration')
# # axs.set_xlabel('Number of Neighbours')
# # plt.savefig("results/time_var_num_of_neighbours_total.png")
#
# # ------------------------------------------------------------
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
#
# # ------------------------------------------------------------
# # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# # axs.plot(contamination_rate_list, duration_list_get_neighbours)
# # axs.title.set_text("Temporal Performance: get_neighbours()")
# # axs.set_ylabel('Duration')
# # axs.set_xlabel('Contamination Rate')
# # plt.savefig("results/time_var_contamination_rate_get_neighbours.png")
# #
# # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# # axs.plot(contamination_rate_list, duration_list_binary_search)
# # axs.title.set_text("Temporal Performance: binary_search()")
# # axs.set_ylabel('Duration')
# # axs.set_xlabel('Contamination Rate')
# # plt.savefig("results/time_var_contamination_rate_binary_search.png")
# #
# # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# # axs.plot(contamination_rate_list, duration_list_total)
# # axs.title.set_text("Temporal Performance: Total")
# # axs.set_ylabel('Duration')
# # axs.set_xlabel('Contamination Rate')
# # plt.savefig("results/time_var_contamination_rate_total.png")
#
#
