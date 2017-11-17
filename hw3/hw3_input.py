import numpy as np
import pandas as pd


def load_train_data(train_path):
    df = pd.read_csv(train_path)
    image = df['feature']
    x_train = np.zeros([len(image), 48 * 48], dtype='float32')
    for i, img in enumerate(image):
        x_train[i] = np.fromstring(img, dtype='float32', sep=' ')

    y_train = df['label'].astype('int').values
    x_train = np.delete(x_train, 59, axis=0)
    y_train = np.delete(y_train, 59, axis=0)

    return x_train.reshape(-1, 48, 48, 1) / 255, y_train


def load_test_data(test_path):
    df = pd.read_csv(test_path)
    image = df['feature']
    x_test = np.zeros([len(image), 48 * 48], dtype='float32')
    for i, img in enumerate(image):
        x_test[i] = np.fromstring(img, dtype='float32', sep=' ')

    return x_test.reshape(-1, 48, 48, 1) / 255
