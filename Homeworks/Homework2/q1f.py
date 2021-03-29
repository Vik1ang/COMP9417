import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def create_dataset(n=1250, nf=2, nr=0, ni=2, random_state=125):
    '''
    generate a new dataset with
    n: total number of samples
    nf: number of features
    nr: number of redundant features (these are linear combinatins of informative features)
    ni: number of informative features (ni + nr = nf must hold)
    random_state: set for reproducibility
    '''
    X, y = make_classification(n_samples=n,
                               n_features=nf,
                               n_redundant=nr,
                               n_informative=ni,
                               random_state=random_state,
                               n_clusters_per_class=2)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size=X.shape)
    X = StandardScaler().fit_transform(X)
    return X, y


def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        plt.title(title)


if __name__ == '__main__':
    data, target = create_dataset(n=2000, nf=20, nr=12, ni=8, random_state=25)
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.5, test_size=0.5, random_state=15)
    X_train_split = np.split(X_train, 10)
    X_test_split = np.split(X_test, 10)
    Y_train_split = np.split(Y_train, 10)
    Y_test_split = np.split(Y_test, 10)
    score = []
    x_axis = [i for i in range(2, 96)]
    test_accuracy = 0
    train_accuracy = 0
    max_auc = 0
    best_k = 0
    for i in range(2, 96):
        k = 0
        temp = []
        for j in range(10):
            new_x = []
            new_y = []
            DecisionTreeClassifierModel = DecisionTreeClassifier(min_samples_leaf=i)
            for m in range(10):
                if m == k:
                    continue
                new_x.append(X_train_split[m])
                new_y.append(Y_train_split[m])
            new_x = np.concatenate(new_x, axis=0)
            new_y = np.concatenate(new_y, axis=0)
            DecisionTreeClassifierModel.fit(new_x, new_y)
            train_predict = DecisionTreeClassifierModel.predict(X_train_split[k])
            ftr, tpr, _ = roc_curve(Y_train_split[k], train_predict)
            temp_auc = auc(ftr, tpr)
            if temp_auc > max_auc:
                max_auc = temp_auc
                train_accuracy = DecisionTreeClassifierModel.score(X_train, Y_train)
                test_accuracy = DecisionTreeClassifierModel.score(X_test, Y_test)
                best_k = i
            temp.append(temp_auc)
            k += 1
        score.append(temp)
        k = 0

    print('min_samples_leaf: ', best_k)
    print('train_accuracy: ', train_accuracy)
    print('test_accuracy: ', test_accuracy)
    plt.boxplot(score)
    plt.xticks(np.arange(2, 96, 1))
    plt.show()
