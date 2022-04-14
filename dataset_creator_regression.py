import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df_ = pd.read_csv("data/3D_spatial_network.csv")
min_max_scaler = preprocessing.MinMaxScaler()
df_[['long', 'lat']] = min_max_scaler.fit_transform(df_[['long', 'lat']])
for iteration in range(30):
    folder_train = "data/3d_road/"
    if not os.path.exists(folder_train):
        os.makedirs(folder_train)

    df = df_.sample(10000).reset_index(drop=True)
    df.to_csv(folder_train + '/' + str(iteration) + '.csv', index=False)
