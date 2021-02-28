import pandas as pd

import numpy as np


# import matplotlib.pyplot as plt

def pre_processing(x):
    x_max = x.max()
    x_min = x.min()
    for i in range(x.size):
        temp = (x[i] - x_min) / (x_max - x_min)
        x[i] = temp
    return x


if __name__ == '__main__':
    df = pd.read_csv("./real_estate.csv")

    # Question1 (a)
    df.dropna(axis=0, inplace=True)
    df.drop(columns=['transactiondate', 'latitude', 'longitude', 'price'], inplace=True)
    # print(df.index)
    # print(df)

    # Question1 (b)
    x_new_age = pre_processing(np.array(df['age']))
    x_new_mrt = pre_processing(np.array(df['nearestMRT']))
    x_new_nCon = pre_processing(np.array(df['nConvenience']))

    x_new_age_mean = x_new_age.mean()
    x_new_mrt_mean = x_new_mrt.mean()
    x_new_nCon_mean = x_new_nCon.mean()

    # Question2
    size = df.index.size
    training_set = df[:int(size / 2)]
    test_set = df[int(size / 2):]

    first_training_row = training_set.loc[0]
    last_training_row = training_set.iloc[-1]
    first_test_row = test_set.iloc[0]
    last_test_row = test_set.iloc[-1]

    # Question3
    w_0 = 0
    w_1 = 1
    w_2 = 2
    w_3 = 3
    y_i = w_0 + w_1 * x_new_age + w_2 * x_new_mrt + w_3 * x_new_nCon