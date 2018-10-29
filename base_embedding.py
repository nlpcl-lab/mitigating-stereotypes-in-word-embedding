#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, codecs, time, re
import gensim
from gensim import utils, matutils
from gensim.models import word2vec, FastText
from gensim.test.utils import datapath
import math
import numpy as np
from collections import OrderedDict
import logging

import config

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Basic constants
DATASET_DIR = 'D:\\PycharmProjects_D\\Bias\\source\\'
MODEL_NAME = 'wiki'  # MODEL_NAME = 'twitter_all' # MODEL_NAME = 'news2018'
MODEL_DIR = 'D:\\PycharmProjects_D\\Bias\\'
DEFAULT_ARGUMENTS_W2V = dict(workers=4, sg=1, size=300, window=5, min_count=5, sample=10^-4, negative=5, seed=1, iter=2)
DEFAULT_ARGUMENTS_FT = dict(**DEFAULT_ARGUMENTS_W2V, min_n=3, max_n=6)

start_time = time.time()


class W2vModel(object):
    def __init__(self):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.w2v_fname = MODEL_DIR + 'w2v_{0}_sg_300_neg5_it2.model'.format(MODEL_NAME)
        self.w2v_model = self.load_w2v_model(self.w2v_fname)
        self.w2v_model.init_sims()                          # for using wv.syn0norm

    def load_w2v_model(self, fname):
        try:
            print('Loading W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            w2v_model = word2vec.Word2Vec.load(fname)
            print(w2v_model)

        except IOError:
            print('No existed model. Training W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            texts = config.WikiCorpus()
            w2v_model = word2vec.Word2Vec(texts, **DEFAULT_ARGUMENTS_W2V)
            # init_sims: reduce memory but cannot continue training (because original vectors are removed.)
            w2v_model.init_sims(replace=True)

            w2v_model.save(fname)  # save model

        print('Success to load W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
        return w2v_model

    def test(self):
        analogy_score, result_list = self.w2v_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
        print("score: {:.2f}".format(analogy_score))
        for result_dict in result_list:
            print("{}: True {} / False {}".format(result_dict['section'], result_dict['correct'][:3], result_dict['incorrect'][:3]))
        similarities = self.w2v_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        print(similarities)


class FtModel(object):
    def __init__(self):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.ft_fname = MODEL_DIR + 'ft_{0}_sg_300_neg5_it2.model'.format(MODEL_NAME)
        self.ft_model = self.load_ft_model(self.ft_fname)

    def load_ft_model(self, fname):
        """
        class FastText(sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None,
        word_ngrams=1, sample=0.001, seed=1, workers=3, min_alpha=0.0001, negative=5, cbow_mean=1, hashfxn=hash, iter=5,
        null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH)
        min_n : int
            Min length of char ngrams to be used for training word representations.
        max_n : int
            Max length of char ngrams to be used for training word representations.
            Set max_n to be lesser than min_n to avoid char ngrams being used.
        word_ngrams : int {1,0}
            If 1, uses enriches word vectors with subword(ngrams) information. If 0, this is equivalent to word2vec.
        bucket : int
            Character ngrams are hashed into a fixed number of buckets, in order to limit the memory usage of the model.
            This option specifies the number of buckets used by the model.
        """
        print('Loading Fasttext Model... in {0:.2f} seconds'.format(time.time() - start_time))
        try:
            fasttext_model = FastText.load(fname)
            print(fasttext_model)
        except IOError:
            print('No existed model. Training Ft Model... in {0:.2f} seconds'.format(time.time() - start_time))
            texts = config.WikiCorpus()
            fasttext_model = FastText(texts, **DEFAULT_ARGUMENTS_FT)
            fasttext_model.save(fname)

        print('Success to load Fasttext Model... in {0:.2f} seconds'.format(time.time() - start_time))
        return fasttext_model

    def test(self):
        analogy_score, result_list = self.ft_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
        print("score: {:.2f}".format(analogy_score))
        for result_dict in result_list:
            print("{}: True {} / False {}".format(result_dict['section'], result_dict['correct'][:3], result_dict['incorrect'][:3]))
        similarities = self.ft_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        print(similarities)


if __name__ == "__main__":
    w2v = W2vModel()
    ft = FtModel()
    w2v.test()