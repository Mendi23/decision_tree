
from sklearn.tree import tree
from utils import get_split_data


def get_over_and_under_fit_score():
    X_train, X_test, Y_train, Y_test = get_split_data("flare.csv", 32)
    overFit_dt = tree.DecisionTreeClassifier(criterion = "entropy")
    overFit_dt.fit(X_train, Y_train)
    underFit_dt = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 2)
    underFit_dt.fit(X_train, Y_train)

    return overFit_dt.score(X_train, Y_train), \
           overFit_dt.score(X_test, Y_test), \
           underFit_dt.score(X_train, Y_train), \
           underFit_dt.score(X_test, Y_test)

res = get_over_and_under_fit_score()
print(f"OverFit results:     trained: {res[0]}, test: {res[1]}")
print(f"UnderFit results:     trained: {res[2]}, test: {res[3]}")
