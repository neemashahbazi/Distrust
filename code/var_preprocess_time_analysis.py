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


num_of_neighbors = 10
contamination_rate = 0.1
duration_list_train = []
duration_list_sort = []
duration_list_total = []
dimension_list = [i for i in range(2, 21, 1)]

for dim in dimension_list:
    print(dim)
    iteration_list_train = []
    iteration_list_sort = []
    iteration_list_total = []

    for iteration in range(5):
        folder_train = "data/day_trading/train/" + str(iteration) + ".csv"
        folder_test = "data/day_trading/test/" + str(iteration) + ".csv"

        df_train = pd.read_csv(folder_train)
        x = df_train.drop('is_profit', axis=1)
        x = x.to_numpy()
        x = x[:, :dim]
        y = df_train.is_profit
        y = y.to_numpy()

        df_test = pd.read_csv(folder_test)
        x_test_scaled = df_test.drop('is_profit', axis=1)
        x_test_scaled = x_test_scaled.to_numpy()
        x_test_scaled = x_test_scaled[:, :dim]
        y_test = df_test.is_profit
        y_test = y_test.to_numpy()

        start = time.time()
        clf = KNN(n_neighbors=num_of_neighbors)
        clf.fit(x)
        end = time.time()
        duration_train = end - start
        iteration_list_train.append(duration_train)
        neighbours = []
        for test_item in x_test_scaled:
            distance, neighbour = clf.get_neighbours(test_item.reshape(1, -1))
            neighbours.append(neighbour)

        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(x_test_scaled)

        f3_liberal = []
        f3_conservative = []
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        start = time.time()
        y_train_scores.sort()
        end = time.time()
        duration_sort = end - start
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
        iteration_list_sort.append(duration_sort)
        duration_total = duration_train + duration_sort
        iteration_list_total.append(duration_total)
    duration_list_train.append(np.mean(iteration_list_train))
    duration_list_sort.append(np.mean(iteration_list_sort))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(dimension_list, duration_list_train, label='Train', color='r')
axs.plot(dimension_list, duration_list_sort, label='Sort', color='b')
axs.plot(dimension_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Preprocessing Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xlabel('Dimension')
plt.savefig("results/pre_time_var_dimension.png")

print("Preprocess Time: Var Dimension")
print("Train:", duration_list_train)
print("Sort:", duration_list_sort)
print("Total:", duration_list_total)

###############################################

num_of_neighbors = 10
duration_list_train = []
duration_list_sort = []
duration_list_total = []
contamination_rate_list = [0.05, 0.1, 0.15, 0.2, 0.25]

for contamination_rate in contamination_rate_list:
    print(contamination_rate)
    iteration_list_train = []
    iteration_list_sort = []
    iteration_list_total = []

    for iteration in range(5):
        folder_train = "data/day_trading/train/" + str(iteration) + ".csv"
        folder_test = "data/day_trading/test/" + str(iteration) + ".csv"

        df_train = pd.read_csv(folder_train)
        x = df_train.drop('is_profit', axis=1)
        x = x.to_numpy()
        x = x[:, :2]
        y = df_train.is_profit
        y = y.to_numpy()

        df_test = pd.read_csv(folder_test)
        x_test_scaled = df_test.drop('is_profit', axis=1)
        x_test_scaled = x_test_scaled.to_numpy()
        x_test_scaled = x_test_scaled[:, :2]
        y_test = df_test.is_profit
        y_test = y_test.to_numpy()

        start = time.time()
        clf = KNN(n_neighbors=num_of_neighbors)
        clf.fit(x)
        end = time.time()
        duration_train = end - start
        iteration_list_train.append(duration_train)
        neighbours = []
        for test_item in x_test_scaled:
            distance, neighbour = clf.get_neighbours(test_item.reshape(1, -1))
            neighbours.append(neighbour)

        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(x_test_scaled)

        f3_liberal = []
        f3_conservative = []
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        start = time.time()
        y_train_scores.sort()
        end = time.time()
        duration_sort = end - start
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
        iteration_list_sort.append(duration_sort)
        duration_total = duration_train + duration_sort
        iteration_list_total.append(duration_total)
    duration_list_train.append(np.mean(iteration_list_train))
    duration_list_sort.append(np.mean(iteration_list_sort))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(contamination_rate_list, duration_list_train, label='Train', color='r')
axs.plot(contamination_rate_list, duration_list_sort, label='Sort', color='b')
axs.plot(contamination_rate_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Preprocess Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xlabel('Contamination Rate')
plt.savefig("results/pre_time_var_contamination_rate.png")

print("Preprocess Time: Var Contamination")
print("Train:", duration_list_train)
print("Sort:", duration_list_sort)
print("Total:", duration_list_total)

contamination_rate = 0.1
duration_list_train = []
duration_list_sort = []
duration_list_total = []
num_of_neighbors_list = [1, 2, 5, 10, 20, 30, 40, 50]

for num_of_neighbors in num_of_neighbors_list:
    print(num_of_neighbors)
    iteration_list_train = []
    iteration_list_sort = []
    iteration_list_total = []

    for iteration in range(5):
        folder_train = "data/day_trading/train/" + str(iteration) + ".csv"
        folder_test = "data/day_trading/test/" + str(iteration) + ".csv"

        df_train = pd.read_csv(folder_train)
        x = df_train.drop('is_profit', axis=1)
        x = x.to_numpy()
        x = x[:, :2]
        y = df_train.is_profit
        y = y.to_numpy()

        df_test = pd.read_csv(folder_test)
        x_test_scaled = df_test.drop('is_profit', axis=1)
        x_test_scaled = x_test_scaled.to_numpy()
        x_test_scaled = x_test_scaled[:, :2]
        y_test = df_test.is_profit
        y_test = y_test.to_numpy()

        start = time.time()
        clf = KNN(n_neighbors=num_of_neighbors)
        clf.fit(x)
        end = time.time()
        duration_train = end - start
        iteration_list_train.append(duration_train)
        neighbours = []
        for test_item in x_test_scaled:
            distance, neighbour = clf.get_neighbours(test_item.reshape(1, -1))
            neighbours.append(neighbour)

        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(x_test_scaled)

        f3_liberal = []
        f3_conservative = []
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        start = time.time()
        y_train_scores.sort()
        end = time.time()
        duration_sort = end - start
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
        iteration_list_sort.append(duration_sort)
        duration_total = duration_train + duration_sort
        iteration_list_total.append(duration_total)
    duration_list_train.append(np.mean(iteration_list_train))
    duration_list_sort.append(np.mean(iteration_list_sort))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(num_of_neighbors_list, duration_list_train, label='Train', color='r')
axs.plot(num_of_neighbors_list, duration_list_sort, label='Sort', color='b')
axs.plot(num_of_neighbors_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Preprocess Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xlabel('Neighborhood Size')
plt.savefig("results/pre_time_var_neighbor.png")

print("Preprocess Time: Var Neighborhood Size")
print("Train:", duration_list_train)
print("Sort:", duration_list_sort)
print("Total:", duration_list_total)

num_of_neighbors = 10
contamination_rate = 0.1
duration_list_train = []
duration_list_sort = []
duration_list_total = []
num_of_points_list = [50, 100, 200, 300, 400, 500, 1000, 5000, 10000, 50000, 100000]

for num_of_points in num_of_points_list:
    print(num_of_points)
    iteration_list_train = []
    iteration_list_sort = []
    iteration_list_total = []

    for iteration in range(5):
        folder_train = "data/day_trading/train/" + str(num_of_points) + "/" + str(iteration) + ".csv"
        folder_test = "data/day_trading/test/" + str(num_of_points) + "/" + str(iteration) + ".csv"

        df_train = pd.read_csv(folder_train)
        x = df_train.drop('is_profit', axis=1)
        x = x.to_numpy()
        x = x[:, :2]
        y = df_train.is_profit
        y = y.to_numpy()

        df_test = pd.read_csv(folder_test)
        x_test_scaled = df_test.drop('is_profit', axis=1)
        x_test_scaled = x_test_scaled.to_numpy()
        x_test_scaled = x_test_scaled[:, :2]
        y_test = df_test.is_profit
        y_test = y_test.to_numpy()

        start = time.time()
        clf = KNN(n_neighbors=num_of_neighbors)
        clf.fit(x)
        end = time.time()
        duration_train = end - start
        iteration_list_train.append(duration_train)
        neighbours = []
        for test_item in x_test_scaled:
            distance, neighbour = clf.get_neighbours(test_item.reshape(1, -1))
            neighbours.append(neighbour)

        y_train_scores = clf.decision_scores_
        y_test_scores = clf.decision_function(x_test_scaled)

        f3_liberal = []
        f3_conservative = []
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        start = time.time()
        y_train_scores.sort()
        end = time.time()
        duration_sort = end - start
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
        iteration_list_sort.append(duration_sort)
        duration_total = duration_train + duration_sort
        iteration_list_total.append(duration_total)
    duration_list_train.append(np.mean(iteration_list_train))
    duration_list_sort.append(np.mean(iteration_list_sort))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(num_of_points_list, duration_list_train, label='Train', color='r')
axs.plot(num_of_points_list, duration_list_sort, label='Sort', color='b')
axs.plot(num_of_points_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Preprocess Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlabel('Number of Points')
plt.savefig("results/pre_time_var_num_of_points.png")

print("Preprocess Time: Var Num of Points")
print("Train:", duration_list_train)
print("Sort:", duration_list_sort)
print("Total:", duration_list_total)


# /Users/nimashahbazi/PycharmProjects/distrust/venv/bin/python /Users/nimashahbazi/PycharmProjects/distrust/var_preprocess_time_analysis.py
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# Preprocess Time: Var Dimension
# Train: [0.031519412994384766, 0.056528472900390626, 0.08935489654541015, 0.1426158905029297, 0.18085765838623047, 0.2337329387664795, 0.28018665313720703, 0.33713841438293457, 0.4221170902252197, 0.5405912876129151, 0.586337947845459, 0.6847045421600342, 0.6115309238433838, 0.6491096973419189, 2.734563779830933, 2.6493332386016846, 2.674545097351074, 2.6724688529968263, 2.6826226234436037]
# Sort: [0.0005527973175048828, 0.0005068302154541016, 0.000514364242553711, 0.0005087852478027344, 0.0005085945129394531, 0.00052642822265625, 0.0005526542663574219, 0.0005597114562988281, 0.0004910469055175781, 0.0005099296569824219, 0.0005120277404785156, 0.0010184764862060547, 0.0005272865295410156, 0.0004942417144775391, 0.0005308151245117188, 0.0005049705505371094, 0.00048766136169433596, 0.000529336929321289, 0.0004930496215820312]
# Total: [0.03207221031188965, 0.057035303115844725, 0.08986926078796387, 0.14312467575073243, 0.1813662528991699, 0.23425936698913574, 0.28073930740356445, 0.3376981258392334, 0.4226081371307373, 0.5411012172698975, 0.5868499755859375, 0.6857230186462402, 0.6120582103729248, 0.6496039390563965, 2.735094594955444, 2.6498382091522217, 2.6750327587127685, 2.6729981899261475, 2.6831156730651857]
# 0.05
# 0.1
# 0.15
# 0.2
# 0.25
# Preprocess Time: Var Contamination
# Train: [0.03012399673461914, 0.02960057258605957, 0.029642152786254882, 0.02956099510192871, 0.02981538772583008]
# Sort: [0.0004942893981933593, 0.00048322677612304686, 0.0004784584045410156, 0.000546121597290039, 0.0004815578460693359]
# Total: [0.0306182861328125, 0.03008379936218262, 0.0301206111907959, 0.03010711669921875, 0.030296945571899415]
# 1
# 2
# 5
# 10
# 20
# 30
# 40
# 50
# Preprocess Time: Var Neighborhood Size
# Train: [0.016654157638549806, 0.01850285530090332, 0.02201075553894043, 0.03029017448425293, 0.04221234321594238, 0.05562844276428223, 0.06967511177062988, 0.08329987525939941]
# Sort: [0.0004803657531738281, 0.0004791259765625, 0.00047631263732910155, 0.0005095958709716797, 0.0004759788513183594, 0.000481414794921875, 0.0004802227020263672, 0.00048160552978515625]
# Total: [0.01713452339172363, 0.01898198127746582, 0.022487068176269533, 0.03079977035522461, 0.04268832206726074, 0.0561098575592041, 0.07015533447265625, 0.08378148078918457]
# 50
# 100
# 200
# 300
# 400
# 500
# 1000
# 5000
# 10000
# 50000
# 100000
# Preprocess Time: Var Num of Points
# Train: [0.0013123035430908203, 0.0013544082641601563, 0.0015522956848144532, 0.0017844676971435548, 0.0023464202880859376, 0.002583217620849609, 0.0036859035491943358, 0.01354513168334961, 0.03029642105102539, 0.18059101104736328, 0.3838259220123291]
# Sort: [9.775161743164062e-06, 1.18255615234375e-05, 1.5115737915039062e-05, 1.8548965454101562e-05, 2.117156982421875e-05, 2.5796890258789063e-05, 4.487037658691406e-05, 0.00023241043090820311, 0.0004922389984130859, 0.0028097152709960936, 0.00594482421875]
# Total: [0.0013220787048339843, 0.0013662338256835937, 0.0015674114227294921, 0.0018030166625976562, 0.002367591857910156, 0.0026090145111083984, 0.00373077392578125, 0.013777542114257812, 0.030788660049438477, 0.18340072631835938, 0.3897707462310791]
#
# Process finished with exit code 0

