from hw4_input import load_vocab, vectorize
from keras.models import model_from_json
import pandas as pd
import numpy as np
import sys


def load_model(filename):
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + '.h5')
    return loaded_model


def predict(models, X_test, predict):

    temp = np.zeros(X_test.shape[0]).astype(float)
    for model in (models):
        result = model.predict(X_test, verbose=1)
        for i in range(len(result)):
            temp[i] += result[i]

    temp = temp / len(models)
    test = np.where(temp > 0.5, 1, 0).astype(int)
    submission = pd.DataFrame({'label': test})
    submission.to_csv(predict, index_label='id')


def load_test_data(filename):
    docs_test = []
    with open(filename, encoding='utf8') as f:
        next(f)
        for i, line in enumerate(f):
            sentence = line[len(line.split(',')[0]) + 1:][:-1]
            docs_test.append(sentence)

    return docs_test


def main():
    vocab = load_vocab()
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    X_test = load_test_data(sys.argv[1])
    max_len = 39
    X_test = vectorize(X_test, word_idx, max_len)
    model1 = load_model('model/lstm128')
    model1.summary()
    model2 = load_model('model/biGru_64')
    model2.summary()
    model3 = load_model('model/bilstm64')
    model3.summary()
    predict([model1, model2, model3], X_test, sys.argv[2])


if __name__ == '__main__':
    main()
