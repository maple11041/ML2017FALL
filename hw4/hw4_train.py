from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense
from keras.layers.core import Dropout
from hw4_input import load_vocab, load_data, vectorize, get_embedding_matrix
import pandas as pd
import numpy as np
import sys
batch_size = 128
nb_epochs = 15
validation_split = 0.2


def build_model(vocab_size, embedding_matrix, max_length):
    model = Sequential()
    e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add((LSTM(128, activation='tanh', dropout=0.2)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model


def save_model(model):
    model_json = model.to_json()
    with open("model/lstm128.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model/lstm128.h5")
    print("Saved model to disk")


def main():
    vocab = load_vocab()
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    vocab_size = len(vocab) + 1
    X_train, Y_train = load_data(sys.argv[1])
    # print (X_train[:10])
    max_len = 39
    X_train = vectorize(X_train, word_idx, max_len)
    embedding_matrix = np.load('word2vec.npy')
    model = build_model(vocab_size, embedding_matrix, max_len)
    model.fit(X_train, Y_train, epochs=nb_epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split, verbose=1)
    save_model(model)


if __name__ == '__main__':
    main()
