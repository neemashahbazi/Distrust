from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import searchsorted, where
from pyod.models.knn import KNN
from scipy.stats import entropy, pointbiserialr
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


num_of_neighbors = 5
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
for iteration in range(1):

    df = pd.read_csv("data/data_banknote_authentication.txt")

    columns_to_scale = ['X1', 'X2', 'X3', 'X4']
    columns_to_encode = []

    ohe = OneHotEncoder(sparse=False)
    ohe.fit(df[columns_to_encode])
    df_ = pd.concat([df.drop(columns_to_encode, 1), pd.DataFrame(ohe.transform(df[columns_to_encode]))],
                    axis=1).reindex()

    clf = LocalOutlierFactor(contamination=0.1, n_neighbors=num_of_neighbors)
    outlier_index = where(clf.fit_predict(df_) == -1)
    df = df_.drop(outlier_index[0], axis=0)
    y = df.Y
    df = df.drop('Y', axis=1)

    outlier_df = df_.drop(df.index)
    y_outlier = outlier_df.Y
    outlier_df = outlier_df.drop('Y', axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_columns = min_max_scaler.fit_transform(df[columns_to_scale])
    columns_to_encode = df.drop(columns_to_scale, axis=1).to_numpy()
    processed_data = np.concatenate([scaled_columns, columns_to_encode], axis=1)

    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=0.80, random_state=1)
    x = X_train
    y = y_train

    outlier_scaled_columns = min_max_scaler.transform(outlier_df[columns_to_scale])
    outlier_processed_data = np.concatenate(
        [outlier_scaled_columns, outlier_df.drop(columns_to_scale, axis=1).to_numpy()], axis=1)
    x_test_scaled = np.concatenate([X_test, outlier_processed_data])
    y_test = np.concatenate([y_test, y_outlier])

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
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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

print('wdt', accuracy_list_conservative_, fpr_list_conservative_, fnr_list_conservative_)

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
plt.savefig("results/effectiveness/adult/bank_note_binned_trust_wdt.png")

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

print('sdt', accuracy_list_liberal_, fpr_list_liberal_, fnr_list_liberal_)

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
plt.savefig("results/effectiveness/adult/bank_note_binned_trust_sdt.png")

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

# bucket 1
# tn 5325 fp 5 fn 45 tp 11
# acc 0.9907166728555514 fpr 0.0009380863039399625 fnr 0.8035714285714286
# ------------------------
# bucket 2
# tn 1645 fp 9 fn 34 tp 5
# acc 0.9746012994683992 fpr 0.005441354292623942 fnr 0.8717948717948718
# ------------------------
# bucket 3
# tn 1513 fp 21 fn 58 tp 29
# acc 0.9512646514497224 fpr 0.013689700130378096 fnr 0.6666666666666666
# ------------------------
# bucket 4
# tn 1111 fp 23 fn 68 tp 72
# acc 0.9285714285714286 fpr 0.020282186948853614 fnr 0.4857142857142857
# ------------------------
# bucket 5
# tn 1437 fp 123 fn 149 tp 481
# acc 0.8757990867579909 fpr 0.07884615384615384 fnr 0.2365079365079365
# ------------------------
# bucket 6
# tn 2247 fp 440 fn 533 tp 1114
# acc 0.7754960775265344 fpr 0.1637513956084853 fnr 0.323618700667881
# ------------------------
# bucket 7
# tn 3128 fp 1083 fn 1456 tp 1649
# acc 0.6529524330235101 fpr 0.2571835668487295 fnr 0.4689210950080515
# ------------------------
# bucket 8
# tn 918 fp 152 fn 138 tp 254
# acc 0.801641586867305 fpr 0.14205607476635515 fnr 0.3520408163265306
# ------------------------
# bucket 9
# tn 736 fp 94 fn 142 tp 153
# acc 0.7902222222222223 fpr 0.11325301204819277 fnr 0.48135593220338985
# ------------------------
# bucket 10
# tn 127 fp 51 fn 53 tp 68
# acc 0.6521739130434783 fpr 0.28651685393258425 fnr 0.4380165289256198
# ------------------------
# liberal
# bucket 1
# tn 16454 fp 1697 fn 2315 tp 3357
# acc 0.8315913193132687 fpr 0.09349347143408077 fnr 0.40814527503526093
# ------------------------
# bucket 2
# tn 853 fp 99 fn 108 tp 140
# acc 0.8275 fpr 0.10399159663865547 fnr 0.43548387096774194
# ------------------------
# bucket 3
# tn 465 fp 82 fn 93 tp 174
# acc 0.785012285012285 fpr 0.14990859232175502 fnr 0.34831460674157305
# ------------------------
# bucket 4
# tn 252 fp 53 fn 85 tp 55
# acc 0.6898876404494382 fpr 0.1737704918032787 fnr 0.6071428571428571
# ------------------------
# bucket 5
# tn 124 fp 40 fn 48 tp 68
# acc 0.6857142857142857 fpr 0.24390243902439024 fnr 0.41379310344827586
# ------------------------
# bucket 6
# tn 39 fp 30 fn 27 tp 42
# acc 0.5869565217391305 fpr 0.43478260869565216 fnr 0.391304347826087
# ------------------------
# WDT: [5386, 1693, 1621, 1274, 2190, 4334, 7316, 1462, 1125, 299]
# SDT: [23823, 1200, 814, 445, 280, 138]
# wdt [0.9905681396212402 0.9759007678676905 0.9524984577421345 0.9266875981161696 0.8828310502283104 0.7747115828334102 0.6559322033898305 0.7950752393980848 0.7984 0.6428093645484949 ; 0.0007129455909943715 0.004111245465538089 0.01225554106910039 0.019047619047619046 0.06576923076923076 0.15638258280610345 0.24103538351935408 0.14504672897196264 0.10481927710843375 0.24719101123595505 ; 0.8392857142857144 0.8717948717948719 0.6689655172413793 0.5128571428571428 0.24444444444444446 0.33770491803278685 0.4838003220611916 0.3683673469387755 0.4738983050847458 0.51900826446281]
# sdt [0.8327918398186626 0.8285 0.7813267813267812 0.7015730337078651 0.6907142857142858 0.5739130434782609 ; 0.0871907883863148 0.10210084033613445 0.15685557586837293 0.15934426229508197 0.21341463414634143 0.3739130434782609 ; 0.42327221438645973 0.4379032258064516 0.3453183520599251 0.6014285714285715 0.4448275862068966 0.4782608695652174]
# Point Biserial Correlation Conservative:  -0.3117885032744172
# Point Biserial Correlation Liberal:  -0.06635273085157892
#
