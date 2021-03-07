import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def pre_processing(x):
    x_max = x.max()
    x_min = x.min()
    for _ in range(x.size):
        _temp = (x[_] - x_min) / (x_max - x_min)
        x[_] = _temp
    return x


def y_model(w0, w1, w2, w3, x1, x2, x3):
    return w0 + w1 * x1 + w2 * x2 + w3 * x3


# def w_t_X_i(w_t, y):
#     return np.dot(w_t, y.T)
#
#
# def derivation(wX, price):
#     return (wX - price) / (2 * np.sqrt(np.square(wX - price)) + 4)


def gradient_descent(learning_rate, mean, w_t):
    w_new = w_t - learning_rate * mean
    return w_new


def mean_loss(n, w_t, x_values, _prices):
    _sum = 0
    for _ in range(n):
        y_i = y_model(w_t[0], w_t[1], w_t[2], w_t[3], x_values[_][0], x_values[_][1], x_values[_][2])
        # _v1 = y_i - _prices[_]
        # _v2 = 2 * np.sqrt(np.square(_v1) + 4)
        # _sum += (_v1 / _v2)
        # _sum += ((y_i - _prices[_]) / (2 * np.sqrt(np.square(y_i - _prices[_]) + 4)))
        _sum += ((y_i - _prices[_]) / (2 * math.sqrt(pow(y_i - _prices[_], 2) + 4)))
    return _sum / n


def loss_achieved(n, w_t, _c, _prices, x_values):
    _sum = 0
    for _ in range(n):
        y_i = y_model(w_t[0], w_t[1], w_t[2], w_t[3], x_values[_][0], x_values[_][1], x_values[_][2])
        # _v1 = np.square(1 / _c)
        # _v2 = np.square(_prices[_] - y_i)
        # _v3 = _v1 * _v2 + 1
        # _v4 = np.sqrt(_v3) - 1
        # _sum += _v4
        # _sum += (np.sqrt(np.square(1 / _c) * np.square(_prices[_] - y_i) + 1) - 1)
        _sum += (math.sqrt(pow(1 / _c, 2) * pow(_prices[_] - y_i, 2) + 1) - 1)
    return _sum / n


if __name__ == '__main__':
    df = pd.read_csv("./real_estate.csv")

    # Question1 (a)
    null_index = df[df.isnull().any(axis=1) == True].index
    df.dropna(axis=0, inplace=True)
    prices = df['price']
    df.drop(columns=['transactiondate', 'latitude', 'longitude', 'price'], inplace=True)

    # Question1 (b)
    x_new_age = pre_processing(np.array(df['age']))
    x_new_mrt = pre_processing(np.array(df['nearestMRT']))
    x_new_nCon = pre_processing(np.array(df['nConvenience']))

    x_new_age_mean = x_new_age.mean()
    x_new_mrt_mean = x_new_mrt.mean()
    x_new_nCon_mean = x_new_nCon.mean()

    # Question2
    x_new = pd.DataFrame(columns=['age', 'nearestMRT', 'nConvenience'])
    x_new['age'] = x_new_age
    x_new['nearestMRT'] = x_new_mrt
    x_new['nConvenience'] = x_new_nCon
    size = x_new.index.size
    training_price = prices[:int(size / 2)]
    test_price = prices[int(size / 2):]
    training_set = x_new[:int(size / 2)]
    test_set = x_new[int(size / 2):]

    first_training_row = training_set.loc[0]
    last_training_row = training_set.iloc[-1]
    first_test_row = test_set.iloc[0]
    last_test_row = test_set.iloc[-1]
    # Question3
    # Question4

    # Question5
    # (a)
    # print(training_set.columns)
    # col_name = list(training_set.columns)
    # col_name.insert(0, 'w0')
    # training_set = training_set.reindex(columns=col_name, fill_value=1)
    training_price = training_price.values
    # print(col_name)
    y_values = training_set.values
    w = np.ones(4)

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    nIter = 400
    alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
    losses = []
    loss = []
    c = 2
    training_size = training_set.index.size
    for i, ax in enumerate(ax.flat):
        w = np.ones(4)
        for index in range(nIter):
            temp = index
            if temp >= training_size:
                temp -= training_size
            loss_mean = 0
            _sum1 = 0
            for _ in range(training_size):
                y_i = w[0] + w[1] * y_values[_][0] + w[2] * y_values[_][1] + w[3] * y_values[_][2]
                # y_i = y_model(w[0], w[1], w[2], w[3], y_values[_][0], y_values[_][1], y_values[_][2])
                _sum1 += ((y_i - training_price[_]) / (2 * np.sqrt(np.square(y_i - training_price[_]) + 4)))
            loss_mean = _sum1 / training_size
            _w_0 = w[0] - 1 * loss_mean * alphas[i]
            _w_1 = w[1] - y_values[temp][0] * loss_mean * alphas[i]
            _w_2 = w[2] - y_values[temp][1] * loss_mean * alphas[i]
            _w_3 = w[3] - y_values[temp][2] * loss_mean * alphas[i]

            w[0] = _w_0
            w[1] = _w_1
            w[2] = _w_2
            w[3] = _w_3
            _sum2 = 0
            res = 0
            for _ in range(training_size):
                y_i = w[0] + w[1] * y_values[_][0] + w[2] * y_values[_][1] + w[3] * y_values[_][2]
                # y_i = y_model(w[0], w[1], w[2], w[3], y_values[_][0], y_values[_][1], y_values[_][2])
                _sum2 += (np.sqrt(np.square(1 / c) * np.square(training_price[_] - y_i) + 1) - 1)
            res = _sum2 / training_size
            # res = loss_achieved(training_size, w, c, training_price, y_values)
            losses.append(res)
        print(w)
        ax.plot(losses)
        losses.clear()
        ax.set_title(f"step size: {alphas[i]}")

    plt.tight_layout()
    plt.show()

    # Question6
    # fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    # epoch_times = 6
    # for i, ax in enumerate(ax.flat):
    #     np.random.shuffle(tr)
    #     for epoch in range(epoch_times):
    #
    #     ax.plot(losses[i])
    #     ax.set_title(f"step size: {alphas[i]}")
    #
    # plt.tight_layout()
    # plt.show()
