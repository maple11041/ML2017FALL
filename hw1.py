# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys


def getTestData(feature1D, feature2D, df2):

    feature = df2[:18][1]
    feature1DNum = len(feature1D) * 9
    feature2DNum = len(feature2D) * 9
    totalfeatures = feature1DNum + feature2DNum
    test = np.zeros(totalfeatures)
    for i in range(int(len(df2) / 18)):
        temp = df2.iloc[i * 18:i * 18 + 18, 2:].transpose()
        temp.columns = feature
        temp['RAINFALL'] = temp['RAINFALL'].apply(lambda x: 0 if x == 'NR' else x)
        for i in temp.columns:
            temp[i] = pd.to_numeric(temp[i])
        test = np.vstack([test, np.append(temp[feature1D].as_matrix().reshape(feature1DNum), temp[feature2D].as_matrix().reshape(feature2DNum)**2)])
    test = test[1:]
    return test


if __name__ == '__main__':
    feature1D = ['PM2.5', 'PM10', 'O3', 'SO2']
    feature2D = ['PM2.5', 'PM10', 'O3', 'SO2']
    df2 = pd.read_csv(sys.argv[1], encoding='big5', header=None)
    X_test = getTestData(feature1D, feature2D, df2)
    w = np.load('model_w.npy')
    b = np.load('model_b.npy')
    Y_test = b + np.dot(X_test, w)
    submission = pd.DataFrame({'id': df2[0].unique(), 'value': Y_test})
    submission.to_csv(sys.argv[2], index=False)
    # print (submission)
