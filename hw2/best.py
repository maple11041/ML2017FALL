import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import sys


def load_data(X_train, Y_train, X_test):
    X_data = pd.read_csv(X_train)
    Y_data = pd.read_csv(Y_train)
    X_test_data = pd.read_csv(X_test)

    return (X_data, Y_data, X_test_data)


def add_feature(X_data, X_test_data, feature2d, feature3d, feature4d):
    for f in feature2d:
        X_data[f + '2'] = X_data[f] ** 2
        X_test_data[f + '2'] = X_test_data[f] ** 2
    for f in feature3d:
        X_data[f + '3'] = X_data[f] ** 3
        X_test_data[f + '3'] = X_test_data[f] ** 3
    for f in feature4d:
        X_data[f + '4'] = X_data[f] ** 4
        X_test_data[f + '4'] = X_test_data[f] ** 4

    return (X_data, X_test_data)


def scaling(X_data, X_test_data):
    feature_sca = X_data.columns
    X_all = pd.concat([X_data, X_test_data])
    mean = np.zeros(len(feature_sca))
    std = np.zeros(len(feature_sca))
    for i, col in enumerate(feature_sca):
        mean[i] = X_all[col].mean()
        std[i] = X_all[col].std()
    for i, col in enumerate(feature_sca):
        X_data[col] = X_data[col].apply(lambda x: (x - mean[i]) / std[i])
        X_test_data[col] = X_test_data[col].apply(lambda x: (x - mean[i]) / std[i])

    return (X_data, X_test_data)


def train(X_train, Y_train):
    model = GradientBoostingClassifier(n_estimators=400, verbose=0, max_depth=3)
    model.fit(X_train, Y_train)
    return model


if __name__ == '__main__':

    feature2d = ['fnlwgt', 'age', 'hours_per_week']
    feature3d = ['fnlwgt']
    feature4d = []
    X_data, Y_data, X_test_data = load_data(sys.argv[1], sys.argv[2], sys.argv[3])

    X_data, X_test_data = add_feature(X_data, X_test_data, feature2d, feature3d, feature4d)
    X_data, X_test_data = scaling(X_data, X_test_data)

    X_train = X_data.values
    X_test = X_test_data.values
    Y_train = Y_data.values
    Y_train = Y_train.reshape(Y_train.shape[0],)

    model = train(X_train, Y_train)
    result = model.predict(X_test)
    submit = pd.DataFrame({'label': result})
    submit.index += 1
    submit.to_csv(sys.argv[4], index_label='id')
