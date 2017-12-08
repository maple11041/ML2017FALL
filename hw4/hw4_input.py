import numpy as np
import pandas as pd
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    # string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string)


def load_vocab():
    vocab = []
    with open('vocab2.txt', encoding='utf8') as f:
        for i, line in enumerate(f):
            vocab.append(line.split(' ')[0])
    vocab.append("<unk>")
    return vocab


def load_data(training):
    docs = []
    labels = []
    with open(training, encoding='utf8') as f:
        for i, line in enumerate(f):
            sentence = line.split(' +++$+++')[1][:-1]
            sen = sentence.split()
            docs.append(' '.join(sen))
            labels.append(int(line.split(' +++$+++')[0]))

    return docs, np.array(labels)


def vectorize(docs, word_idx, max_length):
    ss = []
    for i, sentence in enumerate(docs):
        words = sentence.split(' ')
        ls = max(0, max_length - len(words))
        index = []
        for w in words:
            try:
                index.append(word_idx[w])
            except KeyError:
                index.append(word_idx["<unk>"])
        ss.append([0] * ls + index)
    return np.array(ss)


def get_embedding_matrix(vocab_size, word_idx):

    embeddings_index = dict()
    f = open('/Users/jerry860307/Desktop/MLhw4/vectors.txt', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size, 200))
    # print (embedding_matrix)
    for word, i in word_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
