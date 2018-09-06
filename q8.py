from sklearn.tree import tree
from utils import get_split_data


def print_Decision_Tree_variations(min_sample_leaf = 20):
    X_train, X_test, Y_train, Y_test = get_split_data("flare.csv", 32)

    id3dt = tree.DecisionTreeClassifier(criterion = "entropy")
    id3dt.fit(X_train, Y_train)
    print(f"DecisionTree No Trim: {id3dt.score(X_test, Y_test)}")

    id3dt = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_leaf = 20)
    id3dt.fit(X_train, Y_train)
    print(f"DecisionTree min_samples_leaf={min_sample_leaf}: {id3dt.score(X_test, Y_test)}")

