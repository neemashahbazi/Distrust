import math
from collections import defaultdict

import matplotlib
import mplcursors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import searchsorted
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy.stats import entropy
from scipy.stats import norm
from scipy.stats import pointbiserialr
from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


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


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, shrink=0.75)
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
                     threshold=None, flag=True, **textkw):
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
            if flag:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])

            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def in_radius(c_x, c_y, r, x, y):
    return math.hypot(c_x - x, c_y - y) <= r


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


font = {
    'size': 25}

matplotlib.rc('font', **font)

num_of_points = 1000
# grid = np.linspace(0, 1, 80)
# x_, y_ = np.meshgrid(grid, grid)
# x_test_scaled = np.vstack([x_.ravel(), y_.ravel()]).T
num_of_neighbors = 10
contamination_rate = 0.1

# y_test = []
#
# for i in range(len(x_test_scaled)):
#     if in_radius(0.30, 0.30, 0.22, x_test_scaled[i][0], x_test_scaled[i][1]):
#         y_test.append(0)
#     else:
#         y_test.append(1)

# for shape in range(1, 14, 1):
iteration_accuracy_list_conservative = []
iteration_fnr_list_conservative = []

iteration_accuracy_list_liberal = []
iteration_fnr_list_liberal = []

iteration_fpr_list_liberal = []
iteration_fpr_list_conservative = []

iteration_corr_list_liberal = []
iteration_corr_list_conservative = []

iteration_bucket_size_conservative = []
iteration_bucket_size_liberal = []

for iteration in [12]:
    folder_train = "../data/synthetic/train/" + str(iteration) + ".csv"
    df = pd.read_csv(folder_train)
    x = df.drop('Y', axis=1)
    x = x.to_numpy()
    y = df.Y
    y = y.to_numpy()

    folder_test = "../data/synthetic/test/" + str(iteration) + ".csv"
    df = pd.read_csv(folder_test)
    x_test_scaled = df.drop('Y', axis=1)
    x_test_scaled = x_test_scaled.to_numpy()
    y_test = df.Y
    y_test = y_test.to_numpy()

    # metrics = ['braycurtis', 'canberra', 'chebyshev', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
    metrics =['manhattan']
    for metric in metrics:
        clf = KNN(n_neighbors=num_of_neighbors, metric='minkowski', p=1)
        clf.fit(x)
        distance, neighbours = clf.get_neighbours(x_test_scaled)
        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(x_test_scaled)

        f3_liberal = []
        f3_conservative = []
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        y_train_scores.sort()
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]

        #################################################
        dict_liberal_features = defaultdict(list)
        dict_conservative_features = defaultdict(list)
        dict_liberal_labels = defaultdict(list)
        dict_conservative_labels = defaultdict(list)

        for idx, val in enumerate(y_test_scores):
            labels = [y[i] for i in neighbours[idx]]
            f1 = shannon_entropy(labels)
            query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
            percentile = (mean * query_index) / outlier_mean_index
            z_score = (percentile - mean) / sd
            f2 = norm.cdf(z_score)
            f3_conservative.append(f1 + f2 - (f1 * f2))
            f3_liberal.append(f1 * f2)
            uncertainty.append(f1)
            outlierness.append(f2)

            if int(np.ceil(10 * f1 * f2)) != 0:
                dict_liberal_features[int(np.ceil(10 * f1 * f2))].append(x_test_scaled[idx])
                dict_liberal_labels[int(np.ceil(10 * f1 * f2))].append(y_test[idx])
            else:
                dict_liberal_features[int(np.ceil(10 * f1 * f2)) + 1].append(x_test_scaled[idx])
                dict_liberal_labels[int(np.ceil(10 * f1 * f2)) + 1].append(y_test[idx])

            dict_conservative_features[int(np.ceil(10 * (f1 + f2 - (f1 * f2))))].append(x_test_scaled[idx])
            dict_conservative_labels[int(np.ceil(10 * (f1 + f2 - (f1 * f2))))].append(y_test[idx])
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        minima = np.min(f3_conservative)
        maxima = np.max(f3_conservative)
        normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
        color = []
        for v in f3_conservative:
            color.append(mapper.to_rgba(v))
        scatter = ax.scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                             color=[item for item in color], s=7)
        ax.set_aspect('equal', adjustable='box')
        ax.title.set_text("WDT")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        cursor = mplcursors.cursor(scatter, hover=True)
        plt.savefig("../results/effectiveness/wdt_" + metric + ".png")

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        minima = np.min(f3_liberal)
        maxima = np.max(f3_liberal)
        normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
        color = []
        for v in f3_liberal:
            color.append(mapper.to_rgba(v))
        scatter = ax.scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
                             color=[item for item in color], s=7)
        ax.set_aspect('equal', adjustable='box')
        ax.title.set_text("SDT")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        cursor = mplcursors.cursor(scatter, hover=True)
        plt.savefig("../results/effectiveness/sdt_" + metric + ".png")

#     params = best_lr(x, y)
#     # model = MLPClassifier(max_iter=2000)
#     model = LogisticRegression(**params)
#     # model = SVC()
#     # model = RandomForestClassifier()
#     model.fit(x, y)
#
#     accuracy_list_conservative = []
#     fnr_list_conservative = []
#     fpr_list_conservative = []
#     bucket_size_conservative = []
#
#     for key in sorted(dict_conservative_features):
#         bucket_size_conservative.append(len(dict_conservative_features[key]))
#         y_pred = model.predict(dict_conservative_features[key])
#         y_true = dict_conservative_labels[key]
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#         # acc = (tp + tn) / (tp + fp + fn + tn)
#         if tp == 0 or (fp + fn == 0 and tp == 0):
#             acc = 0
#         else:
#             acc = tp / (tp + 0.5 * (fp + fn))
#             # acc = (tp + tn) / (tp + fp + fn + tn)
#         if fn == 0:
#             fnr = 0
#         else:
#             fnr = fn / (fn + tp)
#         if fp == 0:
#             fpr = 0
#         else:
#             fpr = fp / (fp + tn)
#         accuracy_list_conservative.append(acc)
#         fnr_list_conservative.append(fnr)
#         fpr_list_conservative.append(fpr)
#
#     iteration_fnr_list_conservative.append(fnr_list_conservative)
#     iteration_accuracy_list_conservative.append(accuracy_list_conservative)
#     iteration_fpr_list_conservative.append(fpr_list_conservative)
#     iteration_bucket_size_conservative.append(bucket_size_conservative)
#
#     accuracy_list_liberal = []
#     fnr_list_liberal = []
#     fpr_list_liberal = []
#     bucket_size_liberal = []
#
#     for key in sorted(dict_liberal_features):
#         bucket_size_liberal.append(len(dict_liberal_features[key]))
#         y_pred = model.predict(dict_liberal_features[key])
#         y_true = dict_liberal_labels[key]
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#         # acc = (tp + tn) / (tp + fp + fn + tn)
#         if tp == 0 or (fp + fn == 0 and tp == 0):
#             acc = 0
#         else:
#             acc = tp / (tp + 0.5 * (fp + fn))
#             # acc = (tp + tn) / (tp + fp + fn + tn)
#
#         if tp == 0:
#             tpr = 0
#         else:
#             tpr = tp / (tp + fn)
#
#         if fp == 0:
#             fpr = 0
#         else:
#             fpr = fp / (fp + tn)
#
#         if fn == 0:
#             fnr = 0
#         else:
#             fnr = fn / (fn + tp)
#         accuracy_list_liberal.append(acc)
#         fnr_list_liberal.append(fnr)
#         fpr_list_liberal.append(fpr)
#     iteration_fnr_list_liberal.append(fnr_list_liberal)
#     iteration_accuracy_list_liberal.append(accuracy_list_liberal)
#     iteration_fpr_list_liberal.append(fpr_list_liberal)
#     iteration_bucket_size_liberal.append(bucket_size_liberal)
#
#     y_pred = model.predict(x_test_scaled)
#     prediction = []
#     for i in range(len(x_test_scaled)):
#         if y_pred[i] == y_test[i]:
#             prediction.append(1)
#         else:
#             prediction.append(0)
#
#     corr, pvalue = pointbiserialr(prediction, f3_conservative)
#     iteration_corr_list_conservative.append(corr)
#     corr, pvalue = pointbiserialr(prediction, f3_liberal)
#     iteration_corr_list_liberal.append(corr)
#
# bucket_size_conservative_ = []
# for i in range(10):
#     bucket_size_conservative_.append(int(np.mean([value[i] for value in iteration_bucket_size_conservative])))
# print('WDT:', bucket_size_conservative_)
#
# bucket_size_liberal_ = []
# for i in range(6):
#     bucket_size_liberal_.append(int(np.mean([value[i] for value in iteration_bucket_size_liberal])))
# print('SDT:', bucket_size_liberal_)
#
# accuracy_list_conservative_ = []
# for i in range(10):
#     accuracy_list_conservative_.append(np.mean([value[i] for value in iteration_accuracy_list_conservative]))
#
# fnr_list_conservative_ = []
# for i in range(10):
#     fnr_list_conservative_.append(np.mean([value[i] for value in iteration_fnr_list_conservative]))
#
# fpr_list_conservative_ = []
# for i in range(10):
#     fpr_list_conservative_.append(np.mean([value[i] for value in iteration_fpr_list_conservative]))
#
# print('wdt', accuracy_list_conservative_, fpr_list_conservative_)
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
#                 '0.9-1.0']
# x_axis_tick = np.arange(10)
# y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# ax.bar(x_axis_tick - 0.30, accuracy_list_conservative_, color='b', width=0.30, label='F1')
# ax.bar(x_axis_tick + 0.00, fnr_list_conservative_, color='r', width=0.30, label='FNR')
# ax.bar(x_axis_tick + 0.30, fpr_list_conservative_, color='y', width=0.30, label='FPR')
# ax.legend()
# ax.set_xticks(x_axis_tick)
# ax.set_xticklabels(x_axis_label, rotation=45)
# ax.set_yticks(y_axis_tick)
# ax.title.set_text("Effectiveness of Weak Distrust Measure")
# ax.set_xlabel('Weak Distrust Measure')
# plt.savefig("../results/effectiveness/binned_trust_wdt.png")
#
# fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# minima = np.min(f3_conservative)
# maxima = np.max(f3_conservative)
# normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
# color = []
# for v in f3_conservative:
#     color.append(mapper.to_rgba(v))
# scatter = ax.scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#                      color=[item for item in color], s=7)
# ax.set_aspect('equal', adjustable='box')
# ax.title.set_text("WDT")
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# cursor = mplcursors.cursor(scatter, hover=True)
# plt.savefig("../results/effectiveness/wdt.png")
#
#
# @cursor.connect("add")
# def on_add(sel):
#     sel.annotation.set(text=f3_conservative[sel.target.index])
#
#
# accuracy_list_liberal_ = []
# for i in range(len(iteration_accuracy_list_liberal[0])):
#     accuracy_list_liberal_.append(np.mean([value[i] for value in iteration_accuracy_list_liberal]))
#
# fnr_list_liberal_ = []
# for i in range(len(iteration_fnr_list_liberal[0])):
#     fnr_list_liberal_.append(np.mean([value[i] for value in iteration_fnr_list_liberal]))
#
# fpr_list_liberal_ = []
# for i in range(len(iteration_fpr_list_liberal[0])):
#     fpr_list_liberal_.append(np.mean([value[i] for value in iteration_fpr_list_liberal]))
#
# print('sdt', accuracy_list_liberal_, fpr_list_liberal_)
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
#                 '0.9-1.0']
# x_axis_tick = np.arange(len(accuracy_list_liberal_))
# y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# ax.bar(x_axis_tick - 0.20, accuracy_list_liberal_, color='b', width=0.20, label='F1')
# ax.bar(x_axis_tick + 0.00, fnr_list_liberal_, color='r', width=0.20, label='FNR')
# ax.bar(x_axis_tick + 0.20, fpr_list_liberal_, color='y', width=0.20, label='FPR')
# ax.legend()
# ax.set_xticks(x_axis_tick)
# ax.set_xticklabels(x_axis_label[:len(accuracy_list_liberal_)], rotation=45)
# ax.set_yticks(y_axis_tick)
# ax.title.set_text("Effectiveness of Strong Distrust Measure")
# ax.set_xlabel('Strong Distrust Measure')
# plt.savefig(
#     "../results/effectiveness/binned_trust_sdt.png")
#
# fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# minima = np.min(f3_liberal)
# maxima = np.max(f3_liberal)
# normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
# color = []
# for v in f3_liberal:
#     color.append(mapper.to_rgba(v))
# scatter = ax.scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#                      color=[item for item in color], s=7)
# ax.set_aspect('equal', adjustable='box')
# ax.title.set_text("SDT")
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# cursor = mplcursors.cursor(scatter, hover=True)
# plt.savefig("../results/effectiveness/sdt.png")
#
#
# @cursor.connect("add")
# def on_add(sel):
#     sel.annotation.set(text=f3_conservative[sel.target.index])
#
#
# # =============================================================================
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
# accuracy_list = []
# tpr_list = []
# fpr_list = []
# fnr_list = []
#
# for i in range(index):
#     y_pred = model.predict(grid_feature[i])
#     y_true = grid_label[i]
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#     acc = (tp + tn) / (tp + fp + fn + tn)
#     if tp == 0:
#         tpr = 0
#     else:
#         tpr = tp / (tp + fn)
#
#     if fp == 0:
#         fpr = 0
#     else:
#         fpr = fp / (fp + tn)
#
#     if fn == 0:
#         fnr = 0
#     else:
#         fnr = fn / (fn + tp)
#
#     accuracy_list.append(acc)
#     tpr_list.append(tpr)
#     fpr_list.append(fpr)
#     fnr_list.append(fnr)
#
# x_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
# B = np.reshape(accuracy_list, (len(y_axis), len(x_axis)))
# accuracy_list_ = B.T
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 14))
# im, cbar = heatmap(accuracy_list_, y_axis, x_axis, ax=ax, cmap="RdYlGn")
# texts = annotate_heatmap(im, data=accuracy_list_, valfmt="{x:.2f}")
# ax.invert_yaxis()
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# ax.title.set_text("F1")
# ax.tick_params(axis='both', which='major', labelsize=30, labelbottom=True, bottom=False, top=False, labeltop=False)
# fig.tight_layout()
# plt.savefig("../results/effectiveness/f1.png")
#
# B = np.reshape(fnr_list, (len(y_axis), len(x_axis)))
# fnr_list_ = B.T
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 14))
# im, cbar = heatmap(fnr_list_, y_axis, x_axis, ax=ax, cmap="RdYlGn_r")
# texts = annotate_heatmap(im, data=fnr_list_, valfmt="{x:.2f}", flag=False)
# ax.invert_yaxis()
# ax.title.set_text("FNR")
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# ax.tick_params(axis='both', which='major', labelsize=30, labelbottom=True, bottom=False, top=False, labeltop=False)
# fig.tight_layout()
# plt.savefig("../results/effectiveness/fnr.png")
#
# B = np.reshape(fpr_list, (len(y_axis), len(x_axis)))
# fpr_list_ = B.T
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 14))
# im, cbar = heatmap(fpr_list_, y_axis, x_axis, ax=ax, cmap="RdYlGn_r")
# texts = annotate_heatmap(im, data=fpr_list_, valfmt="{x:.2f}",flag=False)
# ax.invert_yaxis()
# ax.title.set_text("FPR")
# ax.tick_params(axis='both', which='major', labelsize=30, labelbottom=True, bottom=False, top=False, labeltop=False)
# fig.tight_layout()
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.savefig("../results/effectiveness/fpr.png")
#
# fig, axs = plt.subplots(1, 1, figsize=(9, 9))
# colors = np.where(y == 0, 'r', 'b')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# axs.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8, cmap="RdYlBu", edgecolor="white", s=100)
# axs.title.set_text("Data Set")
# axs.set_aspect('equal', adjustable='box')
# # colors = ['#b22140' if x == 0 else '#4b51a3' for x in y_test]
# # axs[1].scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], s=10, c=colors)
# # axs[1].title.set_text("Test Data")
# # axs[1].set_aspect('equal', adjustable='box')
# plt.savefig("../results/effectiveness/train.png")
#
# print("Point Biserial Correlation Conservative: ", np.mean(iteration_corr_list_conservative))
# print("Point Biserial Correlation Liberal: ", np.mean(iteration_corr_list_liberal))

# WDT: [824, 120, 108, 123, 127, 161, 255, 546, 3015, 1116]
# SDT: [4696, 26, 458, 68, 451, 698]
# Point Biserial Correlation Conservative:  -0.1316669696524399
# Point Biserial Correlation Liberal:  -0.1055656569832628
