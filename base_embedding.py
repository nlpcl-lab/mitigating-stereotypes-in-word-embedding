#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, codecs, time, re
import logging


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
