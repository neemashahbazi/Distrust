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
duration_list_get_neighbours = []
duration_list_binary_search = []
duration_list_total = []
dimension_list = [i for i in range(2, 21, 1)]
dimension_list = [i for i in range(2, 21, 1)]


for dim in dimension_list:
    print(dim)
    iteration_list_get_neighbours = []
    iteration_list_binary_search = []
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
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        y_train_scores.sort()
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
            uncertainty.append(f1)
            outlierness.append(f2)
        avg_binary_search = np.mean(duration_list_binary_search_avg)
        iteration_list_binary_search.append(avg_binary_search)
        avg_total = avg_get_neighbour + avg_binary_search
        iteration_list_total.append(avg_total)
    duration_list_get_neighbours.append(np.mean(iteration_list_get_neighbours))
    duration_list_binary_search.append(np.mean(iteration_list_binary_search))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(dimension_list, duration_list_get_neighbours, label='Get Neighbors', color='r')
axs.plot(dimension_list, duration_list_binary_search, label='Binary Search', color='b')
axs.plot(dimension_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xlabel('Dimension')
plt.savefig("results/time_var_dimension.png")

print("Preprocess Time: Var Dimension")
print("Get Neighborhood:", duration_list_get_neighbours)
print("Binary Search:", duration_list_binary_search)
print("Total:", duration_list_total)

###############################################

num_of_neighbors = 10
duration_list_get_neighbours = []
duration_list_binary_search = []
duration_list_total = []
contamination_rate_list = [0.05, 0.1, 0.15, 0.2, 0.25]

for contamination_rate in contamination_rate_list:
    print(contamination_rate)
    iteration_list_get_neighbours = []
    iteration_list_binary_search = []
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
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        y_train_scores.sort()
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
            uncertainty.append(f1)
            outlierness.append(f2)
        avg_binary_search = np.mean(duration_list_binary_search_avg)
        iteration_list_binary_search.append(avg_binary_search)
        avg_total = avg_get_neighbour + avg_binary_search
        iteration_list_total.append(avg_total)
    duration_list_get_neighbours.append(np.mean(iteration_list_get_neighbours))
    duration_list_binary_search.append(np.mean(iteration_list_binary_search))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(contamination_rate_list, duration_list_get_neighbours, label='Get Neighbors', color='r')
axs.plot(contamination_rate_list, duration_list_binary_search, label='Binary Search', color='b')
axs.plot(contamination_rate_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xlabel('Contamination Rate')
plt.savefig("results/time_var_contamination_rate.png")

print("Preprocess Time: Var Contamination Rate")
print("Get Neighborhood:", duration_list_get_neighbours)
print("Binary Search:", duration_list_binary_search)
print("Total:", duration_list_total)


contamination_rate = 0.1
duration_list_get_neighbours = []
duration_list_binary_search = []
duration_list_total = []
num_of_neighbors_list = [1, 2, 5, 10, 20, 30, 40, 50]

for num_of_neighbors in num_of_neighbors_list:
    print(num_of_neighbors)
    iteration_list_get_neighbours = []
    iteration_list_binary_search = []
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
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        y_train_scores.sort()
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
            uncertainty.append(f1)
            outlierness.append(f2)
        avg_binary_search = np.mean(duration_list_binary_search_avg)
        iteration_list_binary_search.append(avg_binary_search)
        avg_total = avg_get_neighbour + avg_binary_search
        iteration_list_total.append(avg_total)
    duration_list_get_neighbours.append(np.mean(iteration_list_get_neighbours))
    duration_list_binary_search.append(np.mean(iteration_list_binary_search))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(num_of_neighbors_list, duration_list_get_neighbours, label='Get Neighbors', color='r')
axs.plot(num_of_neighbors_list, duration_list_binary_search, label='Binary Search', color='b')
axs.plot(num_of_neighbors_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xlabel('Neighborhood Size')
plt.savefig("results/time_var_neighbor.png")

print("Preprocess Time: Var Neighborhood Size")
print("Get Neighborhood:", duration_list_get_neighbours)
print("Binary Search:", duration_list_binary_search)
print("Total:", duration_list_total)

num_of_neighbors = 10
contamination_rate = 0.1
duration_list_get_neighbours = []
duration_list_binary_search = []
duration_list_total = []
num_of_points_list = [50, 100, 200, 300, 400, 500, 1000, 5000, 10000, 50000, 100000]

for num_of_points in num_of_points_list:
    print(num_of_points)
    iteration_list_get_neighbours = []
    iteration_list_binary_search = []
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
        uncertainty = []
        outlierness = []

        mean = 1.0 - contamination_rate
        sd = 0.1
        y_train_scores.sort()
        outlier_mean_index = int(mean * len(y_train_scores))
        outlier_mean = y_train_scores[outlier_mean_index]
        duration_list_binary_search_avg = []
        #################################################
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
            uncertainty.append(f1)
            outlierness.append(f2)
        avg_binary_search = np.mean(duration_list_binary_search_avg)
        iteration_list_binary_search.append(avg_binary_search)
        avg_total = avg_get_neighbour + avg_binary_search
        iteration_list_total.append(avg_total)
    duration_list_get_neighbours.append(np.mean(iteration_list_get_neighbours))
    duration_list_binary_search.append(np.mean(iteration_list_binary_search))
    duration_list_total.append(np.mean(iteration_list_total))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.plot(num_of_points_list, duration_list_get_neighbours, label='Get Neighbors', color='r')
axs.plot(num_of_points_list, duration_list_binary_search, label='Binary Search', color='b')
axs.plot(num_of_points_list, duration_list_total, label='Total', color='k', linestyle='dashed')
axs.title.set_text("Time Efficiency")
axs.set_ylabel('Time')
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlabel('Number of Points')
plt.savefig("results/time_var_num_of_points.png")

print("Preprocess Time: Var Num of Points")
print("Get Neighborhood:", duration_list_get_neighbours)
print("Binary Search:", duration_list_binary_search)
print("Total:", duration_list_total)


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
# Get Neighborhood: [0.0003681861999299791, 0.00034704933431413436, 0.0003527437120013767, 0.0003556913778516981, 0.00035975655025906035, 0.0003651626274320815, 0.0003705480729209052, 0.00038354338275061714, 0.00038518800735473635, 0.00039449754661983924, 0.00040341268963283957, 0.00039354700724283855, 0.00041096362219916456, 0.0004137209971745809, 0.0006892550871107313, 0.0007081849516762627, 0.000721422372394138, 0.0007257558488845825, 0.000745565020773146]
# Binary Search: [7.459327909681531e-06, 7.1299547619289826e-06, 7.079271740383572e-06, 7.0505735609266495e-06, 7.0555252499050564e-06, 7.021334966023763e-06, 7.0618746015760635e-06, 7.0287031597561305e-06, 6.98031054602729e-06, 6.996050940619575e-06, 7.037123574150933e-06, 7.032551765441894e-06, 7.019112375047472e-06, 6.998506122165256e-06, 6.988289621141222e-06, 6.980907652113173e-06, 6.962281333075629e-06, 6.983899010552301e-06, 6.982748773362902e-06]
# Total: [0.0003756455278396607, 0.00035417928907606336, 0.0003598229837417603, 0.00036274195141262477, 0.00036681207550896534, 0.0003721839623981052, 0.00037760994752248126, 0.00039057208591037327, 0.00039216831790076363, 0.00040149359756045874, 0.00041044981320699053, 0.0004005795590082804, 0.00041798273457421196, 0.00042071950329674614, 0.0006962433767318725, 0.000715165859328376, 0.0007283846537272135, 0.0007327397478951349, 0.0007525477695465089]
# 0.05
# 0.1
# 0.15
# 0.2
# 0.25
# Preprocess Time: Var Contamination Rate
# Get Neighborhood: [0.00034361297183566617, 0.0003456733640034994, 0.00034155238310496015, 0.00034127198590172666, 0.0003414452346165975]
# Binary Search: [7.0325549443562835e-06, 7.024567392137315e-06, 7.020508448282879e-06, 6.992047097947862e-06, 6.992626190185548e-06]
# Total: [0.00035064552678002245, 0.00035269793139563663, 0.00034857289155324297, 0.0003482640329996745, 0.000348437860806783]
# 1
# 2
# 5
# 10
# 20
# 30
# 40
# 50
# Preprocess Time: Var Neighborhood Size
# Get Neighborhood: [0.00033913287798563636, 0.0003392875379986233, 0.00034086782561408146, 0.0003420911931991577, 0.00034298797024620906, 0.0003447814506954617, 0.00034668283568488226, 0.00034755257977379695]
# Binary Search: [6.836705737643771e-06, 6.945979860093858e-06, 6.970263587103949e-06, 7.032346725463866e-06, 6.9869968626234264e-06, 7.01851314968533e-06, 7.002557118733724e-06, 7.030252350701225e-06]
# Total: [0.00034596958372328017, 0.0003462335178587172, 0.0003478380892011854, 0.0003491235399246216, 0.00034997496710883244, 0.000351799963845147, 0.000353685392803616, 0.00035458283212449815]
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
# Get Neighborhood: [0.0003384727954864502, 0.0003394478707843357, 0.00034114203823937316, 0.00034017653995090065, 0.00034052033742268883, 0.0003403435050116645, 0.0003400617620680067, 0.0003412349563174778, 0.0003418466626273262, 0.0003452080414030287, 0.00034486719449361165]
# Binary Search: [6.8640963236490885e-06, 6.879302130805122e-06, 6.890158653259278e-06, 6.891844537523057e-06, 6.894226604037815e-06, 6.950533654954697e-06, 6.93531248304579e-06, 7.007396486070421e-06, 6.945336129930284e-06, 7.099616792466905e-06, 7.124181853400337e-06]
# Total: [0.0003453368918100993, 0.0003463271729151407, 0.00034803219689263237, 0.0003470683844884237, 0.00034741456402672663, 0.0003472940386666192, 0.0003469970745510525, 0.00034824235280354817, 0.00034879199875725647, 0.0003523076581954956, 0.00035199137634701194]


