import numpy as np
import pandas as pd


def pre_processing(x_data):
    x_max = x_data.max()
    x_min = x_data.min()
    for _ in range(x_data.size):
        _temp = (x_data[_] - x_min) / (x_max - x_min)
        x_data[_] = _temp
    return x_data


if __name__ == '__main__':
    df = pd.read_csv("./real_estate.csv")
    np.set_printoptions(suppress=True)

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
    # training_price = prices.values[:int(size / 2)]
    # test_price = prices.values[int(size / 2):]
    # training_set = x_new.values[:int(size / 2)]
    # test_set = x_new.values[int(size / 2):]

    x_new.insert(loc=3, column='price', value=prices.values)
    training_set_1 = x_new.values[:int(size / 2)]
    test_set_1 = x_new.values[int(size / 2):]
    first_training_row = training_set_1[0]
    last_training_row = training_set_1[-1]
    first_test_row = test_set_1[0]
    last_test_row = test_set_1[-1]
    print("first training row: ", first_training_row)
    print("last training row: ", last_training_row)
    print("first test row: ", first_test_row)
    print("last test row: ", last_test_row)
