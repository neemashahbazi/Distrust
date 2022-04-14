import numpy as np
from aif360.datasets import CompasDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = CompasDataset().convert_to_dataframe()
df = df[0]
columns_to_encode = ['sex', 'race']
columns_to_scale = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
min_max_scaler = preprocessing.MinMaxScaler()
ohe = OneHotEncoder(sparse=False)
scaled_columns = min_max_scaler.fit_transform(df[columns_to_scale])
encoded_columns = ohe.fit_transform(df[columns_to_encode])
df = df.drop(columns_to_encode, axis=1)
df = df.drop(columns_to_scale, axis=1)

processed_data = np.concatenate([scaled_columns, encoded_columns, df], axis=1)
y_ = df.two_year_recid
y_ = y_.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(processed_data, y_, test_size=0.80, random_state=1)


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
from scipy.stats import entropy
from scipy.stats import norm
from scipy.stats import pointbiserialr
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


num_of_neighbors = 50
contamination_rate = 0.1

iteration_accuracy_list_conservative = []
iteration_fnr_list_conservative = []
iteration_fpr_list_conservative = []

iteration_accuracy_list_liberal = []
iteration_fnr_list_liberal = []
iteration_fpr_list_liberal = []

iteration_corr_list_liberal = []
iteration_corr_list_conservative = []

iteration_bucket_size_conservative = []
iteration_bucket_size_liberal = []

df_ = pd.read_csv("data/credit.csv")
for iteration in range(30):
    df = df_.sample(n=15000).reset_index(drop=True)
    df = df.drop('ID', axis=1)
    columns_to_encode = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']
    columns_to_scale = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
    min_max_scaler = preprocessing.MinMaxScaler()
    ohe = OneHotEncoder(sparse=False)
    scaled_columns = min_max_scaler.fit_transform(df[columns_to_scale])
    encoded_columns = ohe.fit_transform(df[columns_to_encode])

    processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
    # train, test = train_test_split(df, train_size=5000, test_size=10000, random_state=42)
    y_ = df.Y
    y_ = y_.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(processed_data, y_, test_size=0.80, random_state=1)

    x = X_train
    y = y_train

    x_test_scaled = X_test
    y_test = y_test

    clf = KNN(n_neighbors=num_of_neighbors)
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

    model = MLPClassifier(max_iter=1000)
    model.fit(x, y)

    accuracy_list_conservative = []
    fnr_list_conservative = []
    fpr_list_conservative = []
    bucket_size_conservative = []

    for key in sorted(dict_conservative_features):
        bucket_size_conservative.append(len(dict_conservative_features[key]))
        y_pred = model.predict(dict_conservative_features[key])
        y_true = dict_conservative_labels[key]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels=[0, 1]).ravel()
        print("bucket", key)
        print("tn", tn, "fp", fp, "fn", fn, "tp", tp)

        # acc = tp / (tp + 0.5 * (fp + fn))
        acc = (tp + tn) / (tp + fp + fn + tn)

        if fn == 0:
            fnr = 0
        else:
            fnr = fn / (fn + tp)
        if fp == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        print("acc", acc, "fpr", fpr, "fnr", fnr)
        print("------------------------")
        accuracy_list_conservative.append(acc)
        fnr_list_conservative.append(fnr)
        fpr_list_conservative.append(fpr)

    iteration_fnr_list_conservative.append(fnr_list_conservative)
    iteration_fpr_list_conservative.append(fpr_list_conservative)
    iteration_accuracy_list_conservative.append(accuracy_list_conservative)
    iteration_bucket_size_conservative.append(bucket_size_conservative)

    accuracy_list_liberal = []
    fnr_list_liberal = []
    fpr_list_liberal = []
    bucket_size_liberal = []
    print("liberal")
    for key in sorted(dict_liberal_features):
        bucket_size_liberal.append(len(dict_liberal_features[key]))
        y_pred = model.predict(dict_liberal_features[key])
        y_true = dict_liberal_labels[key]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        print("bucket", key)
        print("tn", tn, "fp", fp, "fn", fn, "tp", tp)
        # acc = tp / (tp + 0.5 * (fp + fn))
        acc = (tp + tn) / (tp + fp + fn + tn)
        if fp == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        if fn == 0:
            fnr = 0
        else:
            fnr = fn / (fn + tp)
        print("acc", acc, "fpr", fpr, "fnr", fnr)
        print("------------------------")

        accuracy_list_liberal.append(acc)
        fnr_list_liberal.append(fnr)
        fpr_list_liberal.append(fpr)
    iteration_fnr_list_liberal.append(fnr_list_liberal)
    iteration_fpr_list_liberal.append(fpr_list_liberal)
    iteration_accuracy_list_liberal.append(accuracy_list_liberal)
    iteration_bucket_size_liberal.append(bucket_size_liberal)

    y_pred = model.predict(x_test_scaled)
    prediction = []
    for i in range(len(x_test_scaled)):
        if y_pred[i] == y_test[i]:
            prediction.append(1)
        else:
            prediction.append(0)

    corr, pvalue = pointbiserialr(prediction, f3_conservative)
    iteration_corr_list_conservative.append(corr)
    corr, pvalue = pointbiserialr(prediction, f3_liberal)
    iteration_corr_list_liberal.append(corr)

bucket_size_conservative_ = []
for i in range(10):
    bucket_size_conservative_.append(int(np.mean([value[i] for value in iteration_bucket_size_conservative])))
print('WDT:', bucket_size_conservative_)

bucket_size_liberal_ = []
for i in range(6):
    bucket_size_liberal_.append(int(np.mean([value[i] for value in iteration_bucket_size_liberal])))
print('SDT:', bucket_size_liberal_)

accuracy_list_conservative_ = []
for i in range(10):
    accuracy_list_conservative_.append(np.mean([value[i] for value in iteration_accuracy_list_conservative]))

fnr_list_conservative_ = []
for i in range(10):
    fnr_list_conservative_.append(np.mean([value[i] for value in iteration_fnr_list_conservative]))

fpr_list_conservative_ = []
for i in range(10):
    fpr_list_conservative_.append(np.mean([value[i] for value in iteration_fpr_list_conservative]))

print('wdt', accuracy_list_conservative_, fpr_list_conservative_)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
                '0.9-1.0']
x_axis_tick = np.arange(10)
y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.bar(x_axis_tick - 0.30, accuracy_list_conservative_, color='b', width=0.30, label='F1')
ax.bar(x_axis_tick + 0.00, fnr_list_conservative_, color='r', width=0.30, label='FNR')
ax.bar(x_axis_tick + 0.30, fpr_list_conservative_, color='y', width=0.30, label='FPR')

ax.legend()
ax.set_xticks(x_axis_tick)
ax.set_xticklabels(x_axis_label, rotation=45)
ax.set_yticks(y_axis_tick)
ax.title.set_text("Effectiveness of Weak Distrust Measure")
ax.set_xlabel('Weak Distrust Measure')
plt.savefig("results/effectiveness/credit/binned_trust_wdt.png")

# fig, ax = plt.subplots(1, 1, figsize=(7, 7))
# minima = np.min(f3_conservative)
# maxima = np.max(f3_conservative)
# normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
# color = []
# for v in f3_conservative:
#     color.append(mapper.to_rgba(v))
# scatter = ax.scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#                      color=[item for item in color], s=2)
# ax.set_aspect('equal', adjustable='box')
# ax.title.set_text("Weak Distrust Measure")
# cursor = mplcursors.cursor(scatter, hover=True)
# plt.savefig("results/effectiveness/real_class/wdt.png")
#
#
# @cursor.connect("add")
# def on_add(sel):
#     sel.annotation.set(text=f3_conservative[sel.target.index])


accuracy_list_liberal_ = []
for i in range(len(iteration_accuracy_list_liberal[0])):
    accuracy_list_liberal_.append(np.mean([value[i] for value in iteration_accuracy_list_liberal]))

fnr_list_liberal_ = []
for i in range(len(iteration_fnr_list_liberal[0])):
    fnr_list_liberal_.append(np.mean([value[i] for value in iteration_fnr_list_liberal]))

fpr_list_liberal_ = []
for i in range(len(iteration_fpr_list_liberal[0])):
    fpr_list_liberal_.append(np.mean([value[i] for value in iteration_fpr_list_liberal]))

print('sdt', accuracy_list_liberal_, fpr_list_liberal_)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
                '0.9-1.0']
x_axis_tick = np.arange(len(accuracy_list_liberal_))
y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.bar(x_axis_tick - 0.20, accuracy_list_liberal_, color='b', width=0.20, label='F1')
ax.bar(x_axis_tick + 0.00, fnr_list_liberal_, color='r', width=0.20, label='FNR')
ax.bar(x_axis_tick + 0.20, fpr_list_liberal_, color='y', width=0.20, label='FPR')

ax.legend()
ax.set_xticks(x_axis_tick)
ax.set_xticklabels(x_axis_label[:len(accuracy_list_liberal_)], rotation=45)
ax.set_yticks(y_axis_tick)
ax.title.set_text("Effectiveness of Strong Distrust Measure")
ax.set_xlabel('Strong Distrust Measure')
plt.savefig("results/effectiveness/credit/binned_trust_sdt.png")

# fig, ax = plt.subplots(1, 1, figsize=(7, 7))
# minima = np.min(f3_liberal)
# maxima = np.max(f3_liberal)
# normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
# color = []
# for v in f3_liberal:
#     color.append(mapper.to_rgba(v))
# scatter = ax.scatter([item[0] for item in x_test_scaled], [item[1] for item in x_test_scaled],
#                      color=[item for item in color], s=2)
# ax.set_aspect('equal', adjustable='box')
# ax.title.set_text("Strong Distrust Measure")
# cursor = mplcursors.cursor(scatter, hover=True)
# plt.savefig("results/effectiveness/real_class/sdt.png")
#
#
# @cursor.connect("add")
# def on_add(sel):
#     sel.annotation.set(text=f3_conservative[sel.target.index])


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
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
# im, cbar = heatmap(accuracy_list_, y_axis, x_axis, ax=ax, cbarlabel="Accuracy", cmap="RdYlGn")
# texts = annotate_heatmap(im, data=accuracy_list_, valfmt="{x:.3f}")
# ax.invert_yaxis()
# ax.title.set_text("Accuracy")
# ax.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=False, top=False, labeltop=False)
# fig.tight_layout()
# plt.savefig("results/effectiveness/accuracy.png")
#
# B = np.reshape(fnr_list, (len(y_axis), len(x_axis)))
# fnr_list_ = B.T
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
# im, cbar = heatmap(fnr_list_, y_axis, x_axis, ax=ax, cbarlabel="FNR", cmap="RdYlGn")
# texts = annotate_heatmap(im, data=fnr_list_, valfmt="{x:.3f}")
# ax.invert_yaxis()
# ax.title.set_text("FNR")
# ax.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=False, top=False, labeltop=False)
# fig.tight_layout()
# plt.savefig("results/effectiveness/fnr.png")


fig, axs = plt.subplots(1, 2, figsize=(14, 7))
colors = ['r' if x == 0 else 'b' for x in y]
axs[0].scatter(x[:, 0], x[:, 1], s=10, c=colors)
axs[0].title.set_text("Training Data")
axs[0].set_aspect('equal', adjustable='box')

colors = ['#b22140' if x == 0 else '#4b51a3' for x in y_test]
axs[1].scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], s=10, c=colors)
axs[1].title.set_text("Test Data")
axs[1].set_aspect('equal', adjustable='box')
plt.savefig("results/effectiveness/real_class/train_test.png")

print("Point Biserial Correlation Conservative: ", np.mean(iteration_corr_list_conservative))
print("Point Biserial Correlation Liberal: ", np.mean(iteration_corr_list_liberal))

# Point Biserial Correlation Conservative:  -0.23657808822681148
# Point Biserial Correlation Liberal:  -0.1538442514899607