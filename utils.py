import pandas as pd
from sklearn.model_selection import train_test_split


def get_split_data(path, labelIndex, splitRatio = 0.25):
    labeledData = pd.read_csv(path, sep = ',', index_col = labelIndex, converters = {
        labelIndex: lambda x: 1 if x == "True" else 0 })
    X, Y = labeledData.values[:, :], labeledData.index.values
    return train_test_split(X, Y, test_size = splitRatio)