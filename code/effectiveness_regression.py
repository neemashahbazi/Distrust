from collections import defaultdict

import matplotlib
import mplcursors
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import searchsorted
from pyod.models.knn import KNN
from scipy.stats import norm, pearsonr
from sklearn import preprocessing, tree, metrics
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.3f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_elevation(lat, long):
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    r = requests.get(query).json()
    elevation = pd.io.json.json_normalize(r, 'results')['elevation'].values[0]
    return elevation


iteration_mse_list_conservative = []
iteration_mse_list_liberal = []
iteration_corr_list_liberal = []
iteration_corr_list_conservative = []

iteration_bucket_size_conservative = []
iteration_bucket_size_liberal = []

for iteration in range(30):
    df = pd.read_csv("data/3d_road/" + str(iteration) + ".csv")
    x = df.drop('alt', axis=1)
    y = df.alt
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train_scaled = min_max_scaler.fit_transform(x)

    # grid = np.linspace(0, 1, 80)
    # x_, y_ = np.meshgrid(grid, grid)
    # x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T

    # x_test = min_max_scaler.inverse_transform(x_test_scaled)
    # y_test = []
    # for item in x_test:
    #     altitude = get_elevation(item[1], item[0])
    #     print(item[1], ',', item[0], ':', altitude)
    #     y_test.append(altitude)
    # d = {'long': x_test[:, 0], 'lat': x_test[:, 1], 'alt': y_test}
    # test_data = pd.DataFrame(data=d)
    # test_data.to_csv('data/test_3d_road_network.csv', index=False)

    df = pd.read_csv("data/test_3d_road_network.csv")
    x_test = df.drop('alt', axis=1)
    y_test = df.alt
    x_test_scaled = min_max_scaler.fit_transform(x_test)

    clf = KNN(n_neighbors=10)
    clf.fit(x_train_scaled)
    distance_test, neighbours_test = clf.get_neighbours(x_test_scaled)
    y_test_scores = clf.decision_function(x_test_scaled)
    y_train_scores = clf.decision_scores_

    f3_liberal = []
    f3_conservative = []
    f1 = []
    f2 = []
    # ===================================================
    train_uncertainty_score = []
    distance_train, neighbours_train = clf.get_neighbours(x_train_scaled)
    for idx, val in enumerate(x_train_scaled):
        neighborhood = [y[i] for i in neighbours_train[idx]]
        avg = np.ones(10) * np.mean(neighborhood)
        train_uncertainty_score.append(10 * mean_squared_error(neighborhood, avg))

    mean_uncertainty = 0.90
    sd_uncertainty = 0.1
    train_uncertainty_score.sort()
    uncertainty_mean_index = int(mean_uncertainty * len(train_uncertainty_score))
    uncertainty_mean = train_uncertainty_score[uncertainty_mean_index]

    for idx, val in enumerate(x_test_scaled):
        neighborhood = [y[i] for i in neighbours_test[idx]]
        avg = np.ones(10) * np.mean(neighborhood)
        error = 10 * mean_squared_error(neighborhood, avg)
        query_index_ = searchsorted(train_uncertainty_score, error, side='left', sorter=None)
        percentile_ = (mean_uncertainty * query_index_) / uncertainty_mean_index
        z_score_ = (percentile_ - mean_uncertainty) / sd_uncertainty
        f1.append(norm.cdf(z_score_))

    mean_outlierness = 0.9
    sd_outlierness = 0.1
    y_train_scores.sort()
    outlier_mean_index = int(mean_outlierness * len(y_train_scores))
    outlier_mean = y_train_scores[outlier_mean_index]

    for idx, val in enumerate(y_test_scores):
        query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
        percentile = (mean_outlierness * query_index) / outlier_mean_index
        z_score = (percentile - mean_outlierness) / sd_outlierness
        f2.append(norm.cdf(z_score))

    dict_liberal_features = defaultdict(list)
    dict_conservative_features = defaultdict(list)
    dict_liberal_labels = defaultdict(list)
    dict_conservative_labels = defaultdict(list)

    for i in range(len(x_test_scaled)):
        f3_conservative.append(f1[i] + f2[i] - (f1[i] * f2[i]))
        f3_liberal.append(f1[i] * f2[i])

        if int(np.ceil(10 * f1[i] * f1[i])) != 0:
            dict_liberal_features[int(np.ceil(10 * f1[i] * f1[i]))].append(x_test_scaled[i])
            dict_liberal_labels[int(np.ceil(10 * f1[i] * f1[i]))].append(y_test[i])
        else:
            dict_liberal_features[int(np.ceil(10 * f1[i] * f1[i])) + 1].append(x_test_scaled[i])
            dict_liberal_labels[int(np.ceil(10 * f1[i] * f1[i])) + 1].append(y_test[i])

        dict_conservative_features[int(np.ceil(10 * (f1[i] + f1[i] - (f1[i] * f1[i]))))].append(x_test_scaled[i])
        dict_conservative_labels[int(np.ceil(10 * (f1[i] + f1[i] - (f1[i] * f1[i]))))].append(y_test[i])

    # model = tree.DecisionTreeRegressor()
    # model = ElasticNet()
    model = KNeighborsRegressor()

    model.fit(x_train_scaled, y)
    mse_list = []
    bucket_size_conservative = []
    for key in sorted(dict_conservative_features):
        bucket_size_conservative.append(len(dict_conservative_features[key]))
        y_pred = model.predict(dict_conservative_features[key])
        y_true = dict_conservative_labels[key]
        mse_list.append(metrics.mean_squared_error(y_true, y_pred))
    iteration_mse_list_conservative.append(mse_list)
    iteration_bucket_size_conservative.append(bucket_size_conservative)

    mse_list = []
    bucket_size_liberal = []
    for key in sorted(dict_liberal_features):
        bucket_size_liberal.append(len(dict_liberal_features[key]))
        y_pred = model.predict(dict_liberal_features[key])
        y_true = dict_liberal_labels[key]
        mse_list.append(metrics.mean_squared_error(y_true, y_pred))
    iteration_mse_list_liberal.append(mse_list)
    iteration_bucket_size_liberal.append(bucket_size_liberal)

    y_pred = model.predict(x_test_scaled)
    corr, pvalue = pearsonr(y_pred, f3_conservative)
    iteration_corr_list_conservative.append(corr)
    corr, pvalue = pearsonr(y_pred, f3_liberal)
    iteration_corr_list_liberal.append(corr)

bucket_size_conservative_ = []
for i in range(10):
    bucket_size_conservative_.append(int(np.mean([value[i] for value in iteration_bucket_size_conservative])))
print('WDT:', bucket_size_conservative_)

bucket_size_liberal_ = []
for i in range(8):
    bucket_size_liberal_.append(int(np.mean([value[i] for value in iteration_bucket_size_liberal])))
print('SDT:', bucket_size_liberal_)

mse_list_conservative_ = []
for i in range(10):
    mse_list_conservative_.append(np.mean([value[i] for value in iteration_mse_list_conservative]))

print('mse wdt', mse_list_conservative_)
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
                '0.9-1.0']
x_axis_tick = np.arange(10)
y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.bar(x_axis_tick, mse_list_conservative_, width=0.5, color='b', label='RSS')
ax.set_xticks(x_axis_tick)
ax.legend()
ax.set_xticklabels(x_axis_label, rotation=45)
ax.set_yticks(y_axis_tick)
ax.title.set_text("Effectiveness of Weak Distrust Measure")
ax.set_xlabel('Weak Distrust Measure')
plt.savefig("results/effectiveness/regression/binned_trust_wdt.png")

mse_list_liberal_ = []
for i in range(len(iteration_mse_list_liberal[0])):
    mse_list_liberal_.append(np.mean([value[i] for value in iteration_mse_list_liberal]))

print('mse sdt', mse_list_liberal_)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
                '0.9-1.0']
x_axis_tick = np.arange(len(mse_list_liberal_))
y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.bar(x_axis_tick, mse_list_liberal_, color='b', width=0.5, label='RSS')
ax.legend()
ax.set_xticks(x_axis_tick)
ax.set_xticklabels(x_axis_label[:len(mse_list_liberal_)], rotation=45)
ax.set_yticks(y_axis_tick)
ax.title.set_text("Effectiveness of Strong Distrust Measure")
ax.set_xlabel('Strong Distrust Measure')
plt.savefig("results/effectiveness/regression/binned_trust_sdt.png")
# ===================================================
# # Plot
# minima = np.min(f3_liberal)
# maxima = np.max(f3_liberal)
# norm = Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
#
# color = []
# for v in f3_liberal:
#     color.append(mapper.to_rgba(v))
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# scatter = axs[0].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#                          color=[item for item in color], s=2)
# axs[0].set_aspect('equal', adjustable='box')
# axs[0].title.set_text("Strong Distrust Measure")
# cursor = mplcursors.cursor(scatter, hover=True)
#
#
# @cursor.connect("add")
# def on_add(sel):
#     sel.annotation.set(text=f3_liberal[sel.target.index])
#
#
# minima = np.min(f3_conservative)
# maxima = np.max(f3_conservative)
# norm = Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
# color = []
# for v in f3_conservative:
#     color.append(mapper.to_rgba(v))
# scatter = axs[1].scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#                          color=[item for item in color], s=2)
# axs[1].set_aspect('equal', adjustable='box')
# axs[1].title.set_text("Weak Distrust Measure")
# cursor = mplcursors.cursor(scatter, hover=True)
#
#
# @cursor.connect("add")
# def on_add(sel):
#     sel.annotation.set(text=f3_conservative[sel.target.index])
#
#
# plt.savefig("results/effectiveness/regression/dt.png")
#
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], s=10)
# ax[1].scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], s=10)
#
# plt.savefig("results/effectiveness/regression/train.png")

# =============================================================================
# step = 0.1
# grid_feature = {}
# grid_label = {}
# index = 0
# for i in np.arange(0.0, 1.0, step):
#     for j in np.arange(0.0, 1.0, step):
#         features = []
#         labels = []
#         for k in range(len(x_test_scaled)):
#             if i <= x_test_scaled[k][0] <= i + step and j <= x_test_scaled[k][1] <= j + step:
#                 features.append(x_test_scaled[k])
#                 labels.append(y_test[k])
#             grid_feature.update({index: features})
#             grid_label.update({index: labels})
#         index += 1
#
# mse_list = []
# for i in range(index):
#     y_pred = model.predict(grid_feature[i])
#     y_true = grid_label[i]
#     mse_list.append(metrics.mean_squared_error(y_true, y_pred))
#
# x_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
# B = np.reshape(mse_list, (len(y_axis), len(x_axis)))
# mse_list_ = B.T
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
# im, cbar = heatmap(mse_list_, y_axis, x_axis, ax=ax, cbarlabel="MSE", cmap="RdYlGn_r")
# texts = annotate_heatmap(im, data=mse_list_, valfmt="{x:.2f}")
# ax.invert_yaxis()
# ax.title.set_text("MSE")
# ax.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=False, top=False, labeltop=False)
# fig.tight_layout()
# plt.savefig("results/effectiveness/regression/mse.png")
print("Pearson Correlation Conservative: ", np.mean(iteration_corr_list_conservative))
print("Pearson Correlation Liberal: ", np.mean(iteration_corr_list_liberal))


# Pearson Correlation Conservative:  -0.2582464618775316
# Pearson Correlation Liberal:  0.22762919314120655
