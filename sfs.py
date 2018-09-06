from numpy import average
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from utils import get_split_data


def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score.
    :return: list of chosen feature indexes
    """

    df = DataFrame(x)
    training_df = DataFrame()
    ind = [None]*k

    for i in range(k):
        maxValue = 0

        for j in filter(lambda t: t not in ind, df.columns.values):
            training_df[i] = df[j]
            curScore = score(clf, training_df, y)
            if (curScore > maxValue):
                maxValue = curScore
                ind[i] = j

        training_df[i] = df[ind[i]]
    return ind

def get_knn_acc(k = 8):
    X_train, X_test, Y_train, Y_test = get_split_data("flare.csv", 32)
    X_train, X_test = DataFrame(X_train), DataFrame(X_test)
    clf = KNeighborsClassifier()
    t = sfs(X_train, Y_train, k, clf, score_for_sfs_knn)

    clf.fit(X_train, Y_train)
    print(f"KNN Results, all features: {clf.score(X_test, Y_test)}")

    clf.fit(X_train[t],Y_train)
    print(f"KNN Results, {k} features: {clf.score(X_test[t], Y_test)}")


def score_for_sfs_knn(clf, x, y, kFold = 4):
    cvResults = cross_val_score(clf, x, y, cv = kFold)
    return average(cvResults)


