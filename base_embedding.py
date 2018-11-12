#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, codecs, time, re
import numpy as np
import logging

from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import config

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Basic constants
DATASET_DIR = 'D:\\PycharmProjects_D\\Bias\\source\\'
MODEL_NAME = 'wiki'  # MODEL_NAME = 'twitter_all' # MODEL_NAME = 'news2018'
MY_MODEL_NAME = 'my_embedding'
SEED_NUM = '30'
VOCAB_LIMIT = 100000
ANNOTATED_VOCAB_LIMIT = 10000
MODEL_DIR = 'D:\\PycharmProjects_D\\Bias\\'
neutral_word_list = ['doctor', 'bartender', 'dancer', 'carpenter', 'shopkeeper', 'professor', 'coward', 'entrepreneurs']

DEFAULT_ARGUMENTS_W2V = dict(workers=4, sg=1, size=300, window=5, min_count=5, sample=10^-4, negative=5, seed=1, iter=2)
DEFAULT_ARGUMENTS_FT = dict(**DEFAULT_ARGUMENTS_W2V, min_n=3, max_n=6)

start_time = time.time()

def load_UCI():
    X_train, y_train, X_test, y_test = [], [], [], []
    with codecs.open(DATASET_DIR + 'UCI_adult_dataset.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            line = re.sub(r' ', '', line)
            tokens = re.split(r',', line)
            if len(tokens) == 15:
                X_train.append(tokens[:-1])
                y_train.append(tokens[-1])
    with codecs.open(DATASET_DIR + 'UCI_adult_test.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            if i == 0:
                continue
            line = re.sub(r' ', '', line)
            tokens = re.split(r',', line)
            if len(tokens) == 15:
                X_test.append(tokens[:-1])
                y_test.append(tokens[-1])

    return (X_train, y_train), (X_test, y_test)


def word2rep(_X_train, _y_train, model):
    X_train, y_train = [], []

    for X, y in zip(_X_train, _y_train):
        tmp_X = np.array([])
        for token in X:
            if not re.search(r'[a-zA-Z\-?]+', token):
                tmp_X = np.append(tmp_X, np.array([float(token)/10000]))
                tmp_X = np.append(tmp_X, np.zeros(np.shape(model.syn0[1])[0] - 1))
                #continue
            elif token in model.vocab:
                tmp_X = np.append(tmp_X, model[token])
            # compound with '-': only select first vocab without oov for regulating sizes of all X
            elif re.search(r'-', token):
                add_tokens = re.split(r'-', token)
                i = 1
                for add_token in add_tokens:
                    if add_token in model.vocab:
                        tmp_X = np.append(tmp_X, model[add_token])
                        i = 0
                        break
                    else:
                        continue
                if i:
                    tmp_X = np.append(tmp_X, np.zeros(np.shape(model.syn0[1]), dtype=float))

            else:
                tmp_X = np.append(tmp_X, np.zeros(np.shape(model.syn0[1]), dtype=float))

        if np.shape(tmp_X)[0] > 0:
            X_train.append(tmp_X)
            if re.search(r'>', y):
                y_train.append(1)
            else:
                y_train.append(0)

    return X_train, y_train


class Vocab(object):
    """
    A single vocabulary item, used internally e.g. for constructing binary trees
    (incl. both word leaves and inner nodes).
    Possible Fields:
        - count: how often the word occurred in the training sentences
        - index: the word's index in the embedding
    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


