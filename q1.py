from numpy import reshape
from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate


def get_cross_val_score(classifier, datapath, labelCol, kFold):
    labeledData = pd.read_csv(datapath, sep = ',', index_col = labelCol, converters = {
        labelCol: lambda x: 1 if x == "True" else 0 })
    dataArray = labeledData.values[:, :]
    labelsArray = labeledData.index.values

    def cm(i, j):
        def cm_aux(y_true, y_pred): return confusion_matrix(y_true, y_pred)[i, j]
        return cm_aux

    scoring = { 'tp': make_scorer(cm(1, 1)), 'tn': make_scorer(cm(0, 0)),
                'fp': make_scorer(cm(0, 1)), 'fn': make_scorer(cm(1, 0)) }
    cvResults = cross_validate(classifier, dataArray, labelsArray, scoring = scoring,
                               cv = kFold)

    keys = ("test_" + k for k in ('tp', 'fp', 'fn', 'tn'))
    result = [sum(cvResults[i]) for i in keys]
    return (result[0] + result[3]) / sum(result), result


id3dt = tree.DecisionTreeClassifier(criterion = "entropy")
t = get_cross_val_score(id3dt, "flare.csv", 32, 4)
print(t[0])
print(reshape(t[1],(2,2)))


