import numpy as np
import pandas as pd
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
    X_data['capital'] = X_data['capital_gain'] - X_data['capital_loss']
    X_test_data['capital'] = X_test_data['capital_gain'] - X_test_data['capital_loss']
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


def sigmoid(z):
    z = np.clip(z, -99, 99999999)
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1 - (1e-8))


def train(X_train, Y_train):
    features = X_train.shape[1]
    currLoss = 100000000
    b = 0  # initial b
    w = np.zeros(features)  # initial w
    lr = 0.01  # learning rate
    iteration = 15000
    count = 0
    b_lr = 1e-20
    w_lr = np.zeros(features)
    lamda = 0
    # Iterations

    for i in range(iteration):

        b_grad = 0.0
        w_grad = np.zeros(features)
        z = np.dot(X_train, w) + b
        f = sigmoid(z)

        loss = (-((np.dot(Y_train, np.log(f)) + np.dot((1 - Y_train), np.log(1 - f))))) / X_train.shape[0]

        b_grad = b_grad - np.sum(Y_train - f)
        w_grad = w_grad - np.dot((Y_train - f), X_train)

        b_lr = b_lr + b_grad ** 2
        w_lr = w_lr + w_grad ** 2

        # Update parameters.
        b = b - lr / np.sqrt(b_lr) * b_grad
        w = w - lr / np.sqrt(w_lr) * w_grad

        # if loss > currLoss and count > 2000:
        #     break
        currLoss = loss
        count += 1

    return w, b


if __name__ == '__main__':

    feature2d = ['fnlwgt', 'age', 'hours_per_week']
    feature3d = ['fnlwgt', 'age', 'hours_per_week']
    feature4d = ['fnlwgt', 'age', 'hours_per_week']
    X_data, Y_data, X_test_data = load_data(sys.argv[1], sys.argv[2], sys.argv[3])

    X_data, X_test_data = add_feature(X_data, X_test_data, feature2d, feature3d, feature4d)
    X_data, X_test_data = scaling(X_data, X_test_data)

    X_train = X_data.values
    X_test = X_test_data.values
    Y_train = Y_data.values
    Y_train = Y_train.reshape(Y_train.shape[0],)
    w, b = train(X_train, Y_train)
    # print (sigmoid(np.dot(X_test, w) + b))
    result = np.where(sigmoid(np.dot(X_test, w) + b) > 0.5, 1, 0)
    submit = pd.DataFrame({'label': result})
    submit.index += 1
    submit.to_csv(sys.argv[4], index_label='id')
