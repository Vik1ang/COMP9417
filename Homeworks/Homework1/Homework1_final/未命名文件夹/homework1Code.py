import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pre_processing(x_data):
    x_max = x_data.max()
    x_min = x_data.min()
    for _ in range(x_data.size):
        _temp = (x_data[_] - x_min) / (x_max - x_min)
        x_data[_] = _temp
    return x_data


def gradient_update(_X_k_i, w_t, training_values, prices_values, _size):
    _sum_gd = 0
    for _i in range(_size):
        wX = np.dot(w_t, training_values[_i])
        _sum_gd += ((_X_k_i * (wX - prices_values[_i])) / (2 * np.sqrt((wX - prices_values[_i]) ** 2 + 1)))
    return _sum_gd / _size


def loss_achieved(w_t, training_values, prices_values, _size, hyper):
    _sum_loss = 0
    for _i in range(_size):
        wX = np.dot(w_t.T, training_values[_i])
        _sum_loss += (np.sqrt(1 / (hyper ** 2) * (prices_values[_i] - wX) ** 2 + 1) - 1)
    return _sum_loss / _size


if __name__ == '__main__':
    df = pd.read_csv("./real_estate.csv")

    # 1(a)
    null_index = df[df.isnull().any(axis=1) == True].index
    df.dropna(axis=0, inplace=True)
    prices = df['price']
    df.drop(columns=['transactiondate', 'latitude', 'longitude', 'price'], inplace=True)
    print("null indices: ", null_index.values)

    # 1(b)
    x_new_age = pre_processing(np.array(df['age']))
    x_new_mrt = pre_processing(np.array(df['nearestMRT']))
    x_new_nCon = pre_processing(np.array(df['nConvenience']))

    x_new_age_mean = x_new_age.mean()
    x_new_mrt_mean = x_new_mrt.mean()
    x_new_nCon_mean = x_new_nCon.mean()
    print("x_new_age_mean: ", x_new_age_mean)
    print("x_new_nearestMRT: ", x_new_mrt_mean)
    print("x_new_nConvenience: ", x_new_nCon_mean)

    # 2
    x_new = pd.DataFrame(columns=['age', 'nearestMRT', 'nConvenience'])
    x_new['age'] = x_new_age
    x_new['nearestMRT'] = x_new_mrt
    x_new['nConvenience'] = x_new_nCon
    size = x_new.index.size
    training_price = prices.values[:int(size / 2)]
    test_price = prices.values[int(size / 2):]
    training_set = x_new.values[:int(size / 2)]
    test_set = x_new.values[int(size / 2):]

    first_training_row = training_set[0]
    last_training_row = training_set[-1]
    first_test_row = test_set[0]
    last_test_row = test_set[-1]

    print("first training row: ", first_training_row)
    print("last training row: ", last_training_row)
    print("first test row: ", first_test_row)
    print("last test row: ", last_test_row)

    # 5
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    nIter = 400
    alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
    losses = []
    training_size = training_set.shape[0]
    training_gd_set = np.insert(training_set, 0, 1, axis=1)
    c = 2
    loss = []
    w_plot = []
    for i, ax in enumerate(ax.flat):
        w = np.ones(4)
        for index in range(nIter):
            temp = index
            if temp >= training_size:
                temp -= training_size
            w = w - gradient_update(training_gd_set[temp], w, training_gd_set, training_price, training_size) * alphas[
                i]
            loss_mean = loss_achieved(w, training_gd_set, training_price, training_size, c)
            loss.append(loss_mean)
        # print(w)
        losses.append(loss)
        ax.plot(losses[i])
        loss.clear()
        ax.set_title(f"step size: {alphas[i]}")

    plt.tight_layout()
    plt.show()

    # 5 c
    w_plot.clear()
    fig, ax = plt.subplots(figsize=(10, 10))
    w = np.ones(4)
    w_plot.append(w)
    for index in range(nIter):
        temp = index
        if temp >= training_size:
            temp -= training_size
        Xki = np.array([1, training_gd_set[temp][0], training_gd_set[temp][1], training_gd_set[temp][2]])
        w = w - gradient_update(Xki, w, training_gd_set, training_price, training_size) * 0.3
        w_plot.append(w)
    ax.plot(w_plot)
    plt.show()

    print("w0: ", w[0])
    print("w1: ", w[1])
    print("w2: ", w[2])
    print("w3: ", w[3])

    # y-model
    test_size = test_set.shape[0]
    test_set_gd_set = np.insert(test_set, 0, 1, axis=1)
    test_loss = loss_achieved(w, test_set_gd_set, test_price, test_size, c)
    train_loss = loss_achieved(w, training_gd_set, training_price, training_size, c)
    print("batch training loss: ", train_loss)
    print("batch test_loss: ", test_loss)

    # 6
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    losses.clear()
    loss.clear()
    epoch_times = 6
    for i, ax in enumerate(ax.flat):
        w = np.ones(4)
        for index in range(training_size):
            for _ in range(epoch_times):
                # Xki = np.array([1, training_gd_set[index][0], training_gd_set[index][1], training_gd_set[index][2]])
                Xki = np.array(training_gd_set[index])
                wX = np.dot(w, training_gd_set[index].T)
                derivative_loss = (wX - training_price[index]) / (
                        2 * np.sqrt((wX - training_price[index]) ** 2 + 4))
                w = w - alphas[i] * derivative_loss * Xki
                loss_mean = loss_achieved(w, training_gd_set, training_price, training_size, c)
                loss.append(loss_mean)

        # print(w)
        losses.append(loss)
        ax.plot(losses[i])
        loss.clear()
        ax.set_title(f"step size: {alphas[i]}")

    plt.tight_layout()
    plt.show()

    helper = []
    w = np.ones(4)
    helper.append(w)
    for index in range(training_size):
        for _ in range(epoch_times):
            # Xki = np.array([1, training_gd_set[index][0], training_gd_set[index][1], training_gd_set[index][2]])
            Xki = np.array(training_gd_set[index])
            wX = np.dot(w, training_gd_set[index].T)
            derivative_loss = (wX - training_price[index]) / (
                    2 * np.sqrt((wX - training_price[index]) ** 2 + 4))
            w = w - 0.4 * derivative_loss * Xki
            helper.append(w)
            # loss_mean = loss_achieved(w, training_gd_set, training_price, training_size, c)
            # loss.append(loss_mean)
    plt.plot(helper)
    plt.show()

    print("w0: ", w[0])
    print("w1: ", w[1])
    print("w2: ", w[2])
    print("w3: ", w[3])

    # y-model
    test_loss = loss_achieved(w, test_set_gd_set, test_price, test_size, c)
    train_loss = loss_achieved(w, training_gd_set, training_price, training_size, c)
    print("SGD train_loss: ", train_loss)
    print("SGD test_loss: ", test_loss)
