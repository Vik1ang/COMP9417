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

    # split
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.5, test_size=0.5, random_state=15)

    # classifier
    classified_list = []

    SVCModel = SVC()
    SVCModel.fit(X_train, Y_train)
    classified_list.append(SVCModel)

    LogisticRegressionModel = LogisticRegression()
    LogisticRegressionModel.fit(X_train, Y_train)
    classified_list.append(LogisticRegressionModel)

    AdaBoostClassifierModel = AdaBoostClassifier()
    AdaBoostClassifierModel.fit(X_train, Y_train)
    classified_list.append(AdaBoostClassifierModel)

    RandomForestClassifierModel = RandomForestClassifier()
    RandomForestClassifierModel.fit(X_train, Y_train)
    classified_list.append(RandomForestClassifierModel)

    DecisionTreeClassifierModel = DecisionTreeClassifier()
    DecisionTreeClassifierModel.fit(X_train, Y_train)
    classified_list.append(DecisionTreeClassifierModel)

    MLPClassifierModel = MLPClassifier()
    MLPClassifierModel.fit(X_train, Y_train)
    classified_list.append(MLPClassifierModel)

    classified_name_list = ['SVC', 'LogisticRegression', 'AdaBoostClassifier', 'RandomForestClassifier',
                            'DecisionTreeClassifier', 'MLPClassifier']

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for i, ax in enumerate(ax.flat):
        plotter(classifier=classified_list[i], X=X_train, X_test=X_test, y_test=Y_test, title=classified_name_list[i],
                ax=ax)
    plt.tight_layout()
    plt.show()
