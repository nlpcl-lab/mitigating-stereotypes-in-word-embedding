#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, codecs, time, re
import gensim
from gensim import utils, matutils
from gensim.models import word2vec, FastText
from gensim.test.utils import datapath
import math
import numpy as np
import pandas as pd
from collections import OrderedDict
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

def divide_dataset_by_gender(X_test, y_test, model):
    X_male, y_male, X_female, y_female = [], [], [], []
    for X, y in zip(X_test, y_test):
        #if np.allclose(X[2700:3000], model['Male']):
        if np.allclose(X[1800:2100], model['Male']):
            X_male.append(X)
            y_male.append(y)
        #elif np.allclose(X[2700:3000], model['Female']):
        elif np.allclose(X[1800:2100], model['Female']):
            X_female.append(X)
            y_female.append(y)
        else:
            continue

    return (X_male, y_male), (X_female, y_female)

def print_result(clf, X_male, y_male, normalize=True):
    pred = clf.predict(X_male)
    acc, auc, pre, rec = accuracy_score(y_male, pred), roc_auc_score(y_male, pred), \
                         precision_score(y_male, pred, average=None), recall_score(y_male, pred, average=None)
    cnf_matrix = confusion_matrix(y_male, pred)
    print(acc, auc, pre, rec)
    print(cnf_matrix)
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print(cnf_matrix)
        fpr = cnf_matrix[0, 1]
        fnr = cnf_matrix[1, 0]

    return fpr, fnr


# means predicted:X, target:y
def find_optimal_cutoff(predicted, target):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']).pop()


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


class W2vModel(object):
    def __init__(self, vocab_limit=None):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.w2v_fname = MODEL_DIR + 'w2v_{0}_sg_300_neg5_it2.model'.format(MODEL_NAME)
        self.w2v_model = self.load_w2v_model_new(self.w2v_fname, vocab_limit)
        if not vocab_limit:
            self.w2v_model.init_sims()                          # for using wv.syn0norm

    def load_w2v_model_new(self, fname, vocab_limit):
        try:
            print('Loading W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            w2v_model = word2vec.Word2Vec.load(fname)
            if vocab_limit:    # it uses KeyedVector class (Word2vec.wv). Do not point wv.
                tmp_w2v = gensim.models.KeyedVectors(vector_size=300)
                tmp_w2v.index2word = w2v_model.wv.index2word[:vocab_limit]
                tmp_w2v.vocab = {w: w2v_model.wv.vocab[w] for w in tmp_w2v.index2word}

                # check if the order of keyedvector is broken
                for i, w in enumerate(tmp_w2v.index2word):
                    if tmp_w2v.vocab[w].index != i:
                        print(w, tmp_w2v.vocab[w].index, i)

                tmp_w2v.syn0 = w2v_model.wv.syn0[:vocab_limit, :]
                w2v_model.wv.vocab = {}
                w2v_model.wv.index2word = []
                w2v_model.wv.syn0 = np.zeros((10, 300))
                print(tmp_w2v)
                return tmp_w2v

            print(w2v_model)

        except IOError:
            print('No existed model. Training W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            texts = config.WikiCorpus()
            w2v_model = word2vec.Word2Vec(texts, **DEFAULT_ARGUMENTS_W2V)
            # init_sims: reduce memory but cannot continue training (because original vectors are removed.)
            w2v_model.init_sims(replace=True)

            w2v_model.save(fname)  # save model

        print('Success to load W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))

        return w2v_model.wv

    def test(self):
        try:
            self.w2v_model.wv.accuracy(DATASET_DIR+'questions-words.txt', restrict_vocab=300000)
            """
            analogy_score, result_list = self.w2v_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
            print("score: {:.2f}".format(analogy_score))
            for result_dict in result_list:
                print("{}: True {} / False {}".format(result_dict['section'], result_dict['correct'][:3], result_dict['incorrect'][:3]))
            """
        except Exception as e:
            self.w2v_model.accuracy(DATASET_DIR + 'questions-words.txt', restrict_vocab=300000)
        try:
            similarities = self.w2v_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)
        except Exception as e:
            similarities = self.w2v_model.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)

    def test_UCI(self, uci_dataset, small_train=False, overall_acc=True):
        (_X_train, _y_train), (_X_test, _y_test) = uci_dataset
        (X_train, y_train), (X_test, y_test) = word2rep(_X_train, _y_train, self.w2v_model), word2rep(_X_test, _y_test, self.w2v_model)
        (X_male, y_male), (X_female, y_female) = divide_dataset_by_gender(X_test, y_test, self.w2v_model)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        print("num of tests / num of labels: {} {} / {} {} in {:.2f} sec".format(
            len(X_train), len(X_test), len(set(y_train)), len(set(y_test)), time.time() - start_time))

        for c in [1, 10, 100]:
            clf = svm.SVC(C=c)
            if small_train:
                clf.fit(X_train[:10000], y_train[:10000])
            else:
                clf.fit(X_train, y_train)
            if overall_acc:
                print_result(clf, X_test, y_test)

            male_fpr, male_fnr = print_result(clf, X_male, y_male)
            female_fpr, female_fnr = print_result(clf, X_female, y_female)
            print("fpr_bias_ratio: {:.2f}, fnr_bias_ratio: {:.2f}".format(male_fpr/female_fpr, male_fnr/female_fnr))
            print('-' * 30)

        return 0

    def test_analogy(self):
        for word in neutral_word_list:
            print(self.w2v_model.most_similar(positive=['woman', word], negative=['man'], topn=10))

    def save(self):
        self.w2v_model.wv.save_word2vec_format('C:/Users/JAE4258_NLP/PycharmProjects/socialsent-master/socialsent/data/example_embeddings/glove.6B.100d.txt', binary=False)

    def save_vocab(self):
        """
        Setting 4: remove noun particle / foreign words / digit and gender_specific suffix / prefix.
                After that, only remain the data between upper and lower cut off based on frequency.
        :return:
        """
        with codecs.open(DATASET_DIR + '{}_vocabs.txt'.format(MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            tmp_vocab = OrderedDict()
            tmp_list = []
            for word, vocab_obj in sorted(self.w2v_model.wv.vocab.items(), key=lambda item: -item[1].count):
                if re.search(r'^[a-zA-Z][a-zA-Z0-9]{0,}$', word):
                    tmp_vocab[word] = vocab_obj
                    write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            print("Success to save wiki vocabulary.")

        self.w2v_vocab = tmp_vocab

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
        self.ft_model.wv.accuracy(DATASET_DIR + 'questions-words.txt')
        similarities = self.ft_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        # print(similarities)


class MyModel(object):
    def __init__(self, threshold=None, space_order=[1, 1]):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.my_fname = MODEL_DIR + MY_MODEL_NAME + SEED_NUM
        self.my_model = self.load_w2v_model(self.my_fname)
        self.init_modulate = np.shape(self.my_model.syn0)[1]
        self._modulate_vector_linalg(dim=1, dim2=1)
        self.threshold = threshold
        self.space_order = space_order

    def load_w2v_model(self, fname, arranged_savfile=True):
        try:
            print('Loading My Model... in {0:.2f} seconds'.format(time.time() - start_time))
            if not arranged_savfile:
                w2v_model = gensim.models.KeyedVectors.load(fname)
                wi = {w: i for i, w in enumerate(w2v_model.index2word)}
                w2v_model.vocab = {word: Vocab(count=count, index=wi[word]) for word, count in w2v_model.vocab.items()}
                w2v_model.save_word2vec_format(fname, binary=False)

            my_model = word2vec.Word2VecKeyedVectors.load_word2vec_format(fname, binary=False)

            #my_model = word2vec.Word2Vec.load(fname + 'w2vf')
            print(my_model)

        except IOError:
            print('No existed model. Training My Model... in {0:.2f} seconds'.format(time.time() - start_time))
            print("constructing")
            exit()

        print('Success to load My Model... in {0:.2f} seconds'.format(time.time() - start_time))
        return my_model

    def _modulate_vector_linalg(self, dim=1, dim2=1):
        self.my_model.syn0[:, :dim + dim2] = self.my_model.syn0[:, :dim + dim2] / self.init_modulate

    def modulate_sentiment(self, dim=1, dim2=1, intensity=1):
        assert len(self.space_order) < 3, "please set space_order with type 'list' (e.g. [1, 1])."
        if self.threshold and self.space_order[1] == 1:  # modulate sentiment only for entity words
            self.my_model.syn0[:, :dim] = np.multiply(self.my_model.syn0[:, :dim],
                                                      np.where(self.my_model.syn0[:, dim:dim + dim2] >= (self.threshold / self.init_modulate),
                                                               intensity, 1))
        elif self.threshold and self.space_order[1] == -1:  # modulate sentiment only for entity words
            self.my_model.syn0[:, :dim] = np.multiply(self.my_model.syn0[:, :dim],
                                                      np.where(self.my_model.syn0[:, dim:dim + dim2] <= (self.threshold / self.init_modulate),
                                                               intensity, 1))
        else:  # modulate sentiment for entire words
            self.my_model.syn0[:, :dim] = self.my_model.syn0[:, :dim] * intensity
        self.my_model.syn0norm = (self.my_model.syn0 / np.sqrt((self.my_model.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)
        # self.my_model.init_sims(replace=True)
        #  it makes syn0 and vectors to be also normalized (same as syn0norm and vectors_norm)

    def modulate_all(self, dim=1, dim2=1, intensity=1):
        if intensity < 1:
            assert len(self.space_order) < 3, "please set space_order with type 'list' (e.g. [1, 1])."
            self.my_model.syn0[:, :dim+dim2] = self.my_model.syn0[:, :dim+dim2] * intensity
            # self.my_model.init_sims(replace=True)
            #  it makes syn0 and vectors to be also normalized (same as syn0norm and vectors_norm)
        self.my_model.syn0norm = (
                    self.my_model.syn0 / np.sqrt((self.my_model.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)

    def test(self, uci_dataset, intensity_order=1):
        for i, intensity in enumerate([1, 10, 10, 10]):
            #if i == 0:
            #    continue
            print("Model with intensity 10^{}, threshold {}".format(i*intensity_order, self.threshold))
            self.modulate_sentiment(intensity=intensity**intensity_order)
            self.test_analogy()
            #self.show_topn_affect()
            self.test_UCI(uci_dataset, overall_acc=True)
            self.test_intrinsic()

        self.modulate_sentiment(intensity=0)
        self.test_analogy()
        #self.show_topn_affect()
        self.test_UCI(uci_dataset, overall_acc=True)
        self.test_intrinsic()

        """
        self.modulate_all(intensity=0)
        self.test_analogy()
        #self.show_topn_affect()
        self.test_UCI(uci_dataset, overall_acc=True)
        #self.test_intrinsic()
        """


    def test_intrinsic(self):
        self.my_model.accuracy(DATASET_DIR + 'questions-words.txt', restrict_vocab=300000)
        similarities = self.my_model.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)
        print(similarities)

    def test_analogy(self):
        for word in neutral_word_list:
            print(self.my_model.most_similar(positive=['woman', word], negative=['man'], topn=10))

    def test_UCI(self, uci_dataset, small_train=False, overall_acc=True):
        (_X_train, _y_train), (_X_test, _y_test) = uci_dataset
        (X_train, y_train), (X_test, y_test) = word2rep(_X_train, _y_train, self.my_model), word2rep(_X_test, _y_test,
                                                                                                      self.my_model)
        (X_male, y_male), (X_female, y_female) = divide_dataset_by_gender(X_test, y_test, self.my_model)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        print("num of tests / num of labels: {} {} / {} {} in {:.2f} sec".format(
            len(X_train), len(X_test), len(set(y_train)), len(set(y_test)), time.time() - start_time))

        for c in [1, 10, 50, 100]:
            clf = svm.SVC(C=c)
            if small_train:
                clf.fit(X_train[:10000], y_train[:10000])
            else:
                clf.fit(X_train, y_train)
            if overall_acc:
                print_result(clf, X_test, y_test)

            male_fpr, male_fnr = print_result(clf, X_male, y_male)
            female_fpr, female_fnr = print_result(clf, X_female, y_female)
            print("fpr_bias_ratio: {:.2f}, fnr_bias_ratio: {:.2f}".format(male_fpr / female_fpr, male_fnr / female_fnr))
            print('-' * 30)

        return 0

    def show_topn_affect(self, dim=1, dim2=1, topn=1000):
        sort_index_sum = np.ndarray.flatten(self.my_model.vectors[:, :dim]).argsort()
        sort_index = np.prod(self.my_model.vectors[:, :dim+dim2], axis=1).argsort()
        cond = np.ndarray.flatten(self.my_model.vectors[sort_index, dim:dim+dim2]) >= (
                    self.threshold / self.init_modulate) if self.space_order[1] == 1 else \
            np.ndarray.flatten(self.my_model.vectors[sort_index, dim:dim+dim2]) <= (
                    self.threshold / self.init_modulate)

        print("< top {} positive stereotypes >".format(topn))
        if self.space_order[0] == 1:
            for index in sort_index[cond][:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        else:
            for index in sort_index[cond][:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        print("< top {} negative stereotypes >".format(topn))
        if self.space_order[0] == 1:
            for index in sort_index[cond][:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        else:
            for index in sort_index[cond][:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])

