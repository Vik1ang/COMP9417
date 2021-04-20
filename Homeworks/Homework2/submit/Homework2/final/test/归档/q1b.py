### Question 1

# (a)

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
    # data set
    data, target = create_dataset()

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.8, test_size=0.2, random_state=45)
    x_y_set = np.hstack((X_train, Y_train.reshape(1000, 1)))

    size_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # np.random.shuffle(x_y_set)

    # Decision Tree
    DecisionTreeClassifierModel = DecisionTreeClassifier()
    decision_tree_list = []
    for i in range(len(size_list)):
        np.random.shuffle(x_y_set)
        DecisionTreeClassifierModel.fit(x_y_set[:size_list[i], :2], x_y_set[:size_list[i], 2])
        decision_tree_list.append(DecisionTreeClassifierModel.score(X_test, Y_test))

    # Random Forest
    RandomForestClassifierModel = RandomForestClassifier()
    random_forest_list = []
    for i in range(len(size_list)):
        np.random.shuffle(x_y_set)
        RandomForestClassifierModel.fit(x_y_set[:size_list[i], :2], x_y_set[:size_list[i], 2])
        random_forest_list.append(RandomForestClassifierModel.score(X_test, Y_test))

    # AdaBoost
    AdaBoostClassifierModel = AdaBoostClassifier()
    ada_boost_list = []
    for i in range(len(size_list)):
        np.random.shuffle(x_y_set)
        AdaBoostClassifierModel.fit(x_y_set[:size_list[i], :2], x_y_set[:size_list[i], 2])
        ada_boost_list.append(AdaBoostClassifierModel.score(X_test, Y_test))

    # LogisticRegression
    LogisticRegressionModel = LogisticRegression()
    logistic_regression_list = []
    for i in range(len(size_list)):
        np.random.shuffle(x_y_set)
        LogisticRegressionModel.fit(x_y_set[:size_list[i], :2], x_y_set[:size_list[i], 2])
        logistic_regression_list.append(LogisticRegressionModel.score(X_test, Y_test))

    # MLPClassifier
    MLPClassifierModel = MLPClassifier()
    neural_network_list = []
    for i in range(len(size_list)):
        np.random.shuffle(x_y_set)
        MLPClassifierModel.fit(x_y_set[:size_list[i], :2], x_y_set[:size_list[i], 2])
        neural_network_list.append(MLPClassifierModel.score(X_train, Y_train))

    # SVC
    SVCModel = SVC()
    svc_list = []
    for i in range(len(size_list)):
        # np.random.shuffle(x_y_set)
        SVCModel.fit(x_y_set[:size_list[i], :2], x_y_set[:size_list[i], 2])
        svc_list.append(SVCModel.score(X_test, Y_test))

    plt.plot(size_list, decision_tree_list, label='Decision Tree')
    plt.plot(size_list, random_forest_list, label='Random Forest')
    plt.plot(size_list, ada_boost_list, label='AdaBoost')
    plt.plot(size_list, logistic_regression_list, label='Logistic Regression')
    plt.plot(size_list, neural_network_list, label='Neural Network')
    plt.plot(size_list, svc_list, label='SVM')
    plt.legend()
    plt.show()
