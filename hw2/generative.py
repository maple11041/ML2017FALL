import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import pandas as pd
import sys


def load_data(X_train, Y_train, X_test):
    X_data = pd.read_csv(X_train)
    Y_data = pd.read_csv(Y_train)
    X_test_data = pd.read_csv(X_test)

    return (X_data, Y_data, X_test_data)


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


def train(X_data, Y_data):
    features = X_data.columns
    X_data1 = X_data[Y_data['label'] == 1]
    X_data0 = X_data[Y_data['label'] == 0]
    mu0 = np.zeros(len(features))
    mu1 = np.zeros(len(features))
    for i, col in enumerate(features):
        mu0[i] = X_data0[col].mean()
        mu1[i] = X_data1[col].mean()
    X_data0_val = X_data0.values.astype('float64')
    X_data1_val = X_data1.values.astype('float64')
    N0 = X_data0_val.shape[0]
    N1 = X_data1_val.shape[0]
    sigma0 = np.zeros([len(features), len(features)])
    sigma1 = np.zeros([len(features), len(features)])
    for i in range(N0):
        sigma0 += np.dot(np.transpose([(X_data0_val[i] - mu0)]), [X_data0_val[i] - mu0])
    for i in range(N1):
        sigma1 += np.dot(np.transpose([(X_data1_val[i] - mu1)]), [X_data1_val[i] - mu1])
    sigma0 /= N0
    sigma1 /= N1
    share_sigma = (N0 / (N0 + N1)) * sigma0 + (N1 / (N0 + N1)) * sigma1
    try:
        sigma_inv = inv(share_sigma)
    except:
        sigma_inv = pinv(share_sigma)
    b = -0.5 * np.dot(np.dot([mu0], sigma_inv), mu0) + 0.5 * np.dot(np.dot([mu1], sigma_inv), mu1) + np.log(float(N0 / N1))
    w = np.dot((mu0 - mu1), sigma_inv)

    return (w, b)


if __name__ == '__main__':

    X_data, Y_data, X_test_data = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    w, b = train(X_data, Y_data)
    result = np.where(sigmoid(np.dot(X_test_data.values, w) + b) > 0.5, 0, 1)
    submit = pd.DataFrame({'label': result})
    submit.index += 1
    submit.to_csv(sys.argv[4], index_label='id')
