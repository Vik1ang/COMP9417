import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    # Question4

    # Question5
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    nIter = 400
    alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
    for i, ax in enumerate(ax.flat):
        ax.plot(training_set)
        ax.set_title(f"step size: {alphas[i]}")

    plt.tight_layout()
    plt.show()
