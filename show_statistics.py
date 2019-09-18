#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Evaluation config: line 489-497 for mitigated embedding, p822-824 for original embedding

import json, codecs, time, re, os
import gensim
import logging
import numpy as np
import pandas as pd
from gensim.models import word2vec, FastText
from gensim.test.utils import datapath
from sklearn.decomposition import PCA
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from collections import OrderedDict, defaultdict
from copy import deepcopy

# for visualizing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

import config, mitigating_stereotypes, base_words
from config import SOURCE_DIR

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Basic constants
DEFAULT_ARGUMENTS_W2V = dict(workers=4, sg=1, size=300, window=5, min_count=5, sample=10^-4, negative=5, seed=1, iter=2)
DEFAULT_ARGUMENTS_FT = dict(**DEFAULT_ARGUMENTS_W2V, min_n=3, max_n=6)

SVM_Cs = [10]

### UCI setting ###
SMALL_UCI_NUM = 32561


def load_professions(fname):
    with codecs.open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        professions = json.load(f)
    print('Loaded professions\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return professions


sensitive_pair, neutral_word_list = config.load_analogy_pair(SOURCE_DIR + 'minority_groups.txt')


def load_UCI():
    X_train, y_train, X_test, y_test = [], [], [], []
    with codecs.open(SOURCE_DIR + 'UCI_adult_dataset.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            line = re.sub(r' ', '', line)
            tokens = re.split(r',', line)
            if len(tokens) == 15:
                X_train.append(tokens[:-1])
                y_train.append(tokens[-1])
    with codecs.open(SOURCE_DIR + 'UCI_adult_test.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            if i == 0:
                continue
            line = re.sub(r' ', '', line)
            tokens = re.split(r',', line)
            if len(tokens) == 15:
                X_test.append(tokens[:-1])
                y_test.append(tokens[-1])

    print("### UCI train set statistics ###")
    UCI_stats_by_gender(X_train[:SMALL_UCI_NUM], y_train[:SMALL_UCI_NUM])
    print("### UCI test set statistics ###")
    UCI_stats_by_gender(X_test, y_test)

    return (X_train, y_train), (X_test, y_test)


def word2rep(_X_train, _y_train, model):
    X_train, y_train = [], []
    avg_fnl_weight = np.array([int(e[2]) for e in _X_train]).mean()
    avg_hpw_weight = np.array([int(e[12]) for e in _X_train]).mean()

    for X, y in zip(_X_train, _y_train):
        tmp_X = np.array([])
        for token in X:
            if not re.search(r'[a-zA-Z\-?]+', token):
                #tmp_X = np.append(tmp_X, np.array([float(token)/10000]))
                tmp_X = np.append(tmp_X, np.array([float(token)*avg_hpw_weight/avg_fnl_weight]))
                tmp_X = np.append(tmp_X, np.zeros(np.shape(model.syn0[1])[0] - 1))
            elif not config.CONSIDER_GENDER and (token == 'Male' or token == 'Female'):
                continue
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

    return np.array(X_train), np.array(y_train)


def identify_index_by_gender(X, y):
    stats_dict = {}
    stats_dict['Male'] = []
    stats_dict['Female'] = []
    for i, (tokens, y) in enumerate(zip(X, y)):
        stats_dict[tokens[9]].append(i)

    return np.array(stats_dict['Male']), np.array(stats_dict['Female'])


def identify_index_by_race(X, y):
    stats_dict = {}
    stats_dict['Amer-Indian-Eskimo'] = []
    stats_dict['Asian-Pac-Islander'] = []
    stats_dict['Black'] = []
    stats_dict['White'] = []
    stats_dict['Other'] = []
    for i, (tokens, y) in enumerate(zip(X, y)):
        stats_dict[tokens[8]].append(i)

    return np.array(stats_dict['Amer-Indian-Eskimo']), np.array(stats_dict['Asian-Pac-Islander']), \
           np.array(stats_dict['Black']), np.array(stats_dict['White']), np.array(stats_dict['Other'])


def UCI_stats_by_gender(X, y):
    stats_dict = {}
    stats_dict['Male'] = [0, 0]
    stats_dict['Female'] = [0, 0]
    for tokens, label in zip(X, y):
        stats_dict[tokens[9]][1 if re.search(r'>', label) else 0] += 1

    print("<=50K Male:Female = {:.3f} / {:.3f} ({} / {})".format(stats_dict['Male'][0] / (stats_dict['Male'][0] + stats_dict['Female'][0]),
                                                  stats_dict['Female'][0] / (stats_dict['Male'][0] + stats_dict['Female'][0]),
                                                  stats_dict['Male'][0], stats_dict['Female'][0]))
    print(" >50K Male:Female = {:.3f} / {:.3f} ({} / {})".format(stats_dict['Male'][1] / (stats_dict['Male'][1] + stats_dict['Female'][1]),
                                                  stats_dict['Female'][1] / (stats_dict['Male'][1] + stats_dict['Female'][1]),
                                                  stats_dict['Male'][1], stats_dict['Female'][1]))

    return 0


def print_result(y_test, pred, test_male_index, test_female_index):
    acc, auc, pre, rec = accuracy_score(y_test, pred), roc_auc_score(y_test, pred), \
                         precision_score(y_test, pred, average=None), recall_score(y_test, pred, average=None)
    cnf_matrix = confusion_matrix(y_test, pred)
    male_cnf_matrix = confusion_matrix(y_test[test_male_index], pred[test_male_index])
    female_cnf_matrix = confusion_matrix(y_test[test_female_index], pred[test_female_index])
    print(acc, auc, pre, rec)
    print("<=50K Male:Female = {:.3f} / {:.3f} ({} / {})".format(np.sum(male_cnf_matrix, axis=0)[0] / np.sum(cnf_matrix, axis=0)[0],
                                                  np.sum(female_cnf_matrix, axis=0)[0] / np.sum(cnf_matrix, axis=0)[0],
                                                  np.sum(male_cnf_matrix, axis=0)[0], np.sum(female_cnf_matrix, axis=0)[0]))
    print(" >50K Male:Female = {:.3f} / {:.3f} ({} / {})".format(np.sum(male_cnf_matrix, axis=0)[1] / np.sum(cnf_matrix, axis=0)[1],
                                                  np.sum(female_cnf_matrix, axis=0)[1] / np.sum(cnf_matrix, axis=0)[1],
                                                  np.sum(male_cnf_matrix, axis=0)[1], np.sum(female_cnf_matrix, axis=0)[1]))
    fpr, fnr = print_cnf_matrix(cnf_matrix)
    male_fpr, male_fnr = print_cnf_matrix(male_cnf_matrix)
    female_fpr, female_fnr = print_cnf_matrix(female_cnf_matrix)
    print("fpr_bias_ratio: {:.2f}, fnr_bias_ratio: {:.2f}".format(male_fpr / female_fpr, male_fnr / female_fnr))
    print('-' * 30)

    return fpr, fnr


def print_cnf_matrix(cnf_matrix, normalize=True):
    print(cnf_matrix)
    fpr, fnr = 0, 0
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print(cnf_matrix)
        fpr = cnf_matrix[0, 1]
        fnr = cnf_matrix[1, 0]

    return fpr, fnr


# means predicted:X, target:y
def find_optimal_cutoff(predicted, target):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']).pop()


class W2vModel(object):
    def __init__(self, vocab_limit=None):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.w2v_fname = config.WORD_EMBEDDING_NAME
        self.w2v_model = self.load_w2v_model(self.w2v_fname, vocab_limit)
        if not vocab_limit:
            self.w2v_model.init_sims()                          # for using wv.syn0norm

    def load_w2v_model(self, fname, vocab_limit):
        try:
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

            except Exception as e:
                w2v_model = word2vec.Word2VecKeyedVectors.load_word2vec_format(fname, binary=False)
                if vocab_limit:    # it uses KeyedVector class (Word2vec.wv). Do not point wv.
                    tmp_w2v = gensim.models.KeyedVectors(vector_size=300)
                    tmp_w2v.index2word = w2v_model.index2word[:vocab_limit]
                    tmp_w2v.vocab = {w: w2v_model.vocab[w] for w in tmp_w2v.index2word}

                    # check if the order of keyedvector is broken
                    for i, w in enumerate(tmp_w2v.index2word):
                        if tmp_w2v.vocab[w].index != i:
                            print(w, tmp_w2v.vocab[w].index, i)

                    tmp_w2v.syn0 = w2v_model.syn0[:vocab_limit, :]
                    w2v_model.vocab = {}
                    w2v_model.index2word = []
                    w2v_model.syn0 = np.zeros((10, 300))
                    print(tmp_w2v)
                    print('Success to load W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
                    return tmp_w2v

                print(w2v_model)
                print('Success to load W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
                return w2v_model

        except Exception as e:
            print('No existed model. Training W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            texts = ''
            if config.MODEL_NAME == 'wiki':
                texts = config.WikiCorpus()
            elif config.MODEL_NAME == 'reddit':
                texts = config.RedditCorpus()
            else:
                print("please select corpus for training model.")
                exit(1)
            print('training w2v with {} corpus ... in {:.2f} seconds'.format(config.MODEL_NAME, config.whattime()))
            w2v_model = word2vec.Word2Vec(texts, **DEFAULT_ARGUMENTS_W2V)
            # init_sims: reduce memory but cannot continue training (because original vectors are removed.)
            w2v_model.init_sims(replace=True)

            #w2v_model.save(fname)  # save model
            self.w2v_model.save_word2vec_format(fname, binary=False)

        print('Success to load W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))

        return w2v_model.wv

    def test_intrinsic(self):
        try:
            self.w2v_model.wv.accuracy(SOURCE_DIR+'questions-words.txt', restrict_vocab=300000)
            """
            analogy_score, result_list = self.w2v_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
            print("score: {:.2f}".format(analogy_score))
            for result_dict in result_list:
                print("{}: True {} / False {}".format(result_dict['section'], result_dict['correct'][:3], result_dict['incorrect'][:3]))
            """
        except Exception as e:
            self.w2v_model.accuracy(SOURCE_DIR + 'questions-words.txt', restrict_vocab=300000)
        try:
            similarities = self.w2v_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)
        except Exception as e:
            similarities = self.w2v_model.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)

    def test_UCI(self, uci_dataset, small_train=True):
        (_X_train, _y_train), (_X_test, _y_test) = uci_dataset
        test_male_index, test_female_index = identify_index_by_gender(_X_test, _y_test)
        # test_amer_index, test_asian_index, test_black_index, test_white_index, test_other_index = identify_index_by_race(_X_test, _y_test)
        (X_train, y_train), (X_test, y_test) = word2rep(_X_train, _y_train, self.w2v_model), word2rep(_X_test, _y_test,
                                                                                                      self.w2v_model)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        print("num of tests / num of labels: {} {} / {} {} in {:.2f} sec".format(
            len(X_train), len(X_test), len(set(y_train)), len(set(y_test)), time.time() - start_time))

        for c in SVM_Cs:
            clf = svm.SVC(C=c)
            if small_train:
                clf.fit(X_train[:SMALL_UCI_NUM], y_train[:SMALL_UCI_NUM])
            else:
                clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            if not os.path.exists(SOURCE_DIR + 'pred_UCI'):
                os.makedirs(SOURCE_DIR + 'pred_UCI')
            with codecs.open(SOURCE_DIR + 'pred_UCI/w2v_' + config.MODEL_NAME + str(c) + '_pred.txt', 'w', encoding='utf-8', errors='ignore') as f:
                for tokens, label in zip(_X_test, pred):
                    f.write('\t'.join(tokens) + '\t' + str(label) + '\n')

            print_result(y_test, pred, test_male_index, test_female_index)

        return 0

    def test_analogy(self):
        for w1, w2 in sensitive_pair:
            for word in neutral_word_list:
                try:
                    print('{}:{} = {}:{}'.format(
                        w1, w2, word, self.w2v_model.most_similar(positive=[w2, word], negative=[w1], topn=10)))
                except Exception as e:
                    continue


    def save(self, fname):
        self.w2v_model.save_word2vec_format(fname, binary=False)

    def save_vocab(self):
        """
        Setting 4: remove noun particle / foreign words / digit and gender_specific suffix / prefix.
                After that, only remain the data between upper and lower cut off based on frequency.
        :return:
        """
        with codecs.open(SOURCE_DIR + '{}_vocabs.txt'.format(config.MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            tmp_vocab = OrderedDict()
            for word, vocab_obj in sorted(self.w2v_model.wv.vocab.items(), key=lambda item: -item[1].count):
                if re.search(r'^[a-zA-Z][a-zA-Z0-9]{0,}$', word):
                    tmp_vocab[word] = vocab_obj
                    write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            print("Success to save wiki vocabulary.")

        self.w2v_vocab = tmp_vocab

    def get_keyedvectors(self):
        return self.w2v_model


class FtModel(object):
    def __init__(self):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.ft_fname = config.MODEL_DIR + 'ft_{0}_sg_300_neg5_it2.model'.format(config.MODEL_NAME)
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
        self.ft_model.wv.accuracy(SOURCE_DIR + 'questions-words.txt')
        similarities = self.ft_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        # print(similarities)


class MyModel(object):
    def __init__(self, threshold=None, space_order=[1, 1]):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.my_fname = config.MITIGATED_EMBEDDING_NAME
        self.my_model = self.load_w2v_model(self.my_fname)
        self.init_modulate = np.shape(self.my_model.syn0)[1]
        self._modulate_vector_linalg(dim=1, dim2=1)
        self.threshold = threshold
        self.space_order = space_order
        self.modulated_number = 0

    def load_w2v_model(self, fname, arranged_savfile=True):
        try:
            print('Loading My Model... in {0:.2f} seconds'.format(time.time() - start_time))
            if not arranged_savfile:
                w2v_model = gensim.models.KeyedVectors.load(fname)
                wi = {w: i for i, w in enumerate(w2v_model.index2word)}
                w2v_model.vocab = {word: config.Vocab(count=count, index=wi[word]) for word, count in w2v_model.vocab.items()}
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
                                                      np.where(self.my_model.syn0[:, dim:dim + dim2] <= -(self.threshold / self.init_modulate),
                                                               intensity, 1))
        else:  # modulate sentiment for entire words
            self.my_model.syn0[:, :dim] = self.my_model.syn0[:, :dim] * intensity
        self.my_model.syn0norm = (self.my_model.syn0 / np.sqrt((self.my_model.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)
        self.modulated_number += intensity*1
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
        for i, intensity in enumerate([1, 10]): #, 10, 10]):
            if i == 0 and intensity_order < 0:
                continue
            print("Model with intensity 10^{}, threshold {}".format(i*intensity_order, self.threshold))
            self.modulate_sentiment(intensity=intensity**intensity_order)
            self.test_analogy()
            #self.test_UCI(uci_dataset)
            #self.test_intrinsic()
            #self.show_vocab_tsnescatterplot()
            #self.show_topn_embedding()
        print("Model with intensity 0, threshold {}".format(self.threshold))
        self.modulate_sentiment(intensity=0)
        #self.test_analogy()
        self.test_UCI(uci_dataset)
        self.test_intrinsic()

    def test_intrinsic(self):
        self.my_model.accuracy(SOURCE_DIR + 'questions-words.txt', restrict_vocab=300000)
        similarities = self.my_model.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)
        print(similarities)

    def test_analogy(self):
        for w1, w2 in sensitive_pair:
            for word in neutral_word_list:
                try:
                    print('{}:{} = {}:{}'.format(
                        w1, w2, word, self.my_model.most_similar(positive=[w2, word], negative=[w1], topn=10)))
                except Exception as e:
                    continue

    def test_UCI(self, uci_dataset, small_train=True):
        (_X_train, _y_train), (_X_test, _y_test) = uci_dataset
        test_male_index, test_female_index = identify_index_by_gender(_X_test, _y_test)
        (X_train, y_train), (X_test, y_test) = word2rep(_X_train, _y_train, self.my_model), word2rep(_X_test, _y_test,
                                                                                                      self.my_model)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        print("num of tests / num of labels: {} {} / {} {} in {:.2f} sec".format(
            len(X_train), len(X_test), len(set(y_train)), len(set(y_test)), time.time() - start_time))

        for c in SVM_Cs:
            clf = svm.SVC(C=c)
            if small_train:
                clf.fit(X_train[:SMALL_UCI_NUM], y_train[:SMALL_UCI_NUM])
            else:
                clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            with codecs.open(SOURCE_DIR + 'pred_UCI\\my' + str(self.modulated_number) + '_' + config.MODEL_NAME + str(c) + '_pred.txt', 'w', encoding='utf-8', errors='ignore') as f:
                for tokens, label in zip(_X_test, pred):
                    f.write('\t'.join(tokens) + '\t' + str(label) + '\n')
            print_result(y_test, pred, test_male_index, test_female_index)

        return 0

    def show_topn_affect(self, dim=1, dim2=1, topn=50):
        sort_index_sum = np.ndarray.flatten(self.my_model.vectors[:, :dim]).argsort()
        sort_index = np.prod(self.my_model.vectors[:, :dim+dim2], axis=1).argsort()
        cond = np.ndarray.flatten(self.my_model.vectors[sort_index, dim:dim+dim2]) >= (
                    self.threshold / self.init_modulate) if self.space_order[1] == 1 else \
            np.ndarray.flatten(self.my_model.vectors[sort_index, dim:dim+dim2]) <= -(
                    self.threshold / self.init_modulate)

        print("< top {} positive stereotypes >".format(topn))
        if self.space_order[0] * self.space_order[1] == 1:
            for index in sort_index[cond][:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        else:
            for index in sort_index[cond][:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        print("< top {} negative stereotypes >".format(topn))
        if self.space_order[0] * self.space_order[1] == 1:
            for index in sort_index[cond][:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        else:
            for index in sort_index[cond][:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])

    def show_vocab_tsnescatterplot(self, dim=1, dim2=1, shown_word=60, top=False):
        sort_index = np.prod(self.my_model.vectors[:, :dim + dim2], axis=1).argsort()
        cond = np.ndarray.flatten(self.my_model.vectors[sort_index, dim:dim + dim2]) >= (
                self.threshold / self.init_modulate) if self.space_order[1] == 1 else \
            np.ndarray.flatten(self.my_model.vectors[sort_index, dim:dim + dim2]) <= -(
                    self.threshold / self.init_modulate)
        # get random words
        # close_words = model.similar_by_word(word)
        if top:
            entity_words = list(sort_index[cond][::self.space_order[1]])[:int(shown_word / 2)]
            notity_words = list(sort_index[np.logical_not(cond)][::-self.space_order[1]])[:int(shown_word / 2)]
        else:
            entity_words = random.sample(list(sort_index[cond]), int(shown_word / 2))
            notity_words = random.sample(list(sort_index[np.logical_not(cond)]), int(shown_word / 2))

        # add the vector for each of the closest words to the array
        arr, word_labels = np.empty((0, 300), dtype='f'), []
        for index in entity_words + notity_words:
            wrd_vector = self.my_model.syn0norm[index]
            word_labels.append(self.my_model.index2word[index])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

        # find tsne coords for 1 dimensions
        tsne = TSNE(n_components=1, random_state=0)
        np.set_printoptions(suppress=True)

        x_coords = arr[:, 1]
        y_coords = arr[:, 0]
        # display scatter plot
        plt.scatter(x_coords, y_coords)

        for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min() + 0.05, x_coords.max() + 0.05)
        plt.ylim(y_coords.min() + 0.05, y_coords.max() + 0.05)
        plt.show()

    def show_topn_embedding(self, dim=1, dim2=1, topn=30):
        sort_index_sent = np.sum(self.my_model.vectors[:, :dim], axis=1).argsort()
        if self.space_order[0] == -1:
            print("< top {} positive stereotypes >".format(topn))
            for index in sort_index_sent[:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
            print("< top {} negative stereotypes >".format(topn))
            for index in sort_index_sent[:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        else:
            print("< top {} positive stereotypes >".format(topn))
            for index in sort_index_sent[:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
            print("< top {} negative stereotypes >".format(topn))
            for index in sort_index_sent[:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])

        sort_index_sent = np.sum(self.my_model.vectors[:, dim:dim+dim2], axis=1).argsort()
        if self.space_order[1] == -1:
            print("< top {} entity stereotypes >".format(topn))
            for index in sort_index_sent[:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
            print("< top {} notity stereotypes >".format(topn))
            for index in sort_index_sent[:-1-topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim+dim2])
        else:
            print("< top {} entity stereotypes >".format(topn))
            for index in sort_index_sent[:-1 - topn:-1]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim + dim2])
            print("< top {} notity stereotypes >".format(topn))
            for index in sort_index_sent[:topn]:
                print(self.my_model.index2word[index], self.my_model.vectors[index][:dim + dim2])

class DebiasModel(object):
    def __init__(self, bias_model, same_env=True):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        print("same_env: {}".format(same_env))
        if same_env:
            self.model = self.debias_we_same_env(bias_model)
        else:
            self.model = self.debias_we(bias_model)

    def debias_we(self, E):
        print('Loading Debias Model... in {0:.2f} seconds'.format(time.time() - start_time))
        with open(SOURCE_DIR + 'definitional_pairs.json', "r") as f:
            definitional = json.load(f)
        with open(SOURCE_DIR + 'equalize_pairs.json', "r") as f:
            equalize = json.load(f)
        with open(SOURCE_DIR + 'gender_specific_seed.json', "r") as f:
            gender_specific_words = json.load(f)

        tmp_w2v = gensim.models.KeyedVectors(vector_size=300)
        tmp_w2v.index2word = E.index2word
        tmp_w2v.vocab = E.vocab
        tmp_w2v.syn0 = E.syn0
        tmp_w2v.syn0norm = (E.syn0 / np.sqrt((E.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)

        gender_direction = self.doPCA(definitional, tmp_w2v).components_[0]
        specific_set = set(gender_specific_words)
        for i, w in enumerate(tmp_w2v.index2word):
            if w not in specific_set:
                tmp_w2v.syn0[i] = self.drop(tmp_w2v.syn0[i], gender_direction)
        tmp_w2v.syn0norm = (tmp_w2v.syn0 / np.sqrt((tmp_w2v.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)
        candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                         (e1.title(), e2.title()),
                                                         (e1.upper(), e2.upper())]}
        print(candidates)
        for (a, b) in candidates:
            if (a in tmp_w2v.index2word and b in tmp_w2v.index2word):
                y = self.drop((tmp_w2v[a] + tmp_w2v[b]) / 2, gender_direction)
                z = np.sqrt(1 - np.linalg.norm(y) ** 2)
                if (tmp_w2v[a] - tmp_w2v[b]).dot(gender_direction) < 0:
                    z = -z
                tmp_w2v.syn0[tmp_w2v.vocab[a].index] = z * gender_direction + y
                tmp_w2v.syn0[tmp_w2v.vocab[b].index] = -z * gender_direction + y

        tmp_w2v.syn0norm = (tmp_w2v.syn0 / np.sqrt((tmp_w2v.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)

        print('Success to load Debias Model... in {0:.2f} seconds'.format(time.time() - start_time))
        return tmp_w2v

    def debias_we_same_env(self, E, random_sent_pair=False):
        print('Loading Debias (same env.) Model... in {0:.2f} seconds'.format(time.time() - start_time))
        print('example: {} \n {}'.format(np.array(E['Male']), np.array(E['Female'])))
        lexicon = config.load_sent_lexicon()
        lexicon2, lexicon2_vocab = config.load_entity_lexicon()
        num = int(config.BASE_WORD_NUM)
        if random_sent_pair:
            positive_seeds, negative_seeds = mitigating_stereotypes.generate_random_seeds(lexicon, num=num)
        else:
            positive_seeds, negative_seeds = base_words.sent_seeds(10)
        print(positive_seeds, negative_seeds)
        entity_seeds, notity_seeds = mitigating_stereotypes.generate_random_seeds(lexicon2, num=num)
        definitional = zip(positive_seeds, negative_seeds)
        random_pos_seeds, random_neg_seeds = mitigating_stereotypes.generate_random_seeds(lexicon, num=num)
        equalize = zip(random_pos_seeds, random_neg_seeds)
        #notity_specific_words = notity_seeds
        notity_specific_words = [item[0] for item in lexicon2.items() if item[1] == -1]

        tmp_w2v = gensim.models.KeyedVectors(vector_size=300)
        tmp_w2v.index2word = E.index2word
        tmp_w2v.vocab = E.vocab
        tmp_w2v.syn0 = E.syn0
        tmp_w2v.syn0norm = (E.syn0 / np.sqrt((E.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)

        gender_direction = self.doPCA(definitional, tmp_w2v).components_[0]
        specific_set = set(notity_specific_words)
        for i, w in enumerate(tmp_w2v.index2word):
            if w not in specific_set:
                tmp_w2v.syn0[i] = self.drop(tmp_w2v.syn0[i], gender_direction)
        tmp_w2v.syn0norm = (tmp_w2v.syn0 / np.sqrt((tmp_w2v.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)
        candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                         (e1.title(), e2.title()),
                                                         (e1.upper(), e2.upper())]}
        print(candidates)
        for (a, b) in candidates:
            if (a in tmp_w2v.index2word and b in tmp_w2v.index2word):
                y = self.drop((tmp_w2v[a] + tmp_w2v[b]) / 2, gender_direction)
                z = np.sqrt(1 - np.linalg.norm(y) ** 2)
                if (tmp_w2v[a] - tmp_w2v[b]).dot(gender_direction) < 0:
                    z = -z
                tmp_w2v.syn0[tmp_w2v.vocab[a].index] = z * gender_direction + y
                tmp_w2v.syn0[tmp_w2v.vocab[b].index] = -z * gender_direction + y

        tmp_w2v.syn0norm = (tmp_w2v.syn0 / np.sqrt((tmp_w2v.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(float)


        print('Success to load Debias (same env.) Model... in {0:.2f} seconds'.format(time.time() - start_time))
        print('example: {} \n {}'.format(np.array(E['Male']), np.array(E['Female'])))
        return tmp_w2v

    def doPCA(self, pairs, embedding, num_components=10):
        matrix = []
        for a, b in pairs:
            if a in embedding.index2word and b in embedding.index2word:
                center = (embedding[a] + embedding[b]) / 2
                matrix.append(embedding[a] - center)
                matrix.append(embedding[b] - center)
        matrix = np.array(matrix)
        pca = PCA(n_components=num_components)
        pca.fit(matrix)
        # bar(range(num_components), pca.explained_variance_ratio_)
        return pca

    def drop(self, u, v):
        return u - v * u.dot(v) / v.dot(v)

    def test_intrinsic(self):
        try:
            self.model.wv.accuracy(SOURCE_DIR +'questions-words.txt', restrict_vocab=300000)
            """
            analogy_score, result_list = self.w2v_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
            print("score: {:.2f}".format(analogy_score))
            for result_dict in result_list:
                print("{}: True {} / False {}".format(result_dict['section'], result_dict['correct'][:3], result_dict['incorrect'][:3]))
            """
        except Exception as e:
            self.model.accuracy(SOURCE_DIR + 'questions-words.txt', restrict_vocab=300000)
        try:
            similarities = self.model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)
        except Exception as e:
            similarities = self.model.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)

    def test_UCI(self, uci_dataset, small_train=True):
        (_X_train, _y_train), (_X_test, _y_test) = uci_dataset
        test_male_index, test_female_index = identify_index_by_gender(_X_test, _y_test)
        #test_amer_index, test_asian_index, test_black_index, test_white_index, test_other_index = identify_index_by_race(_X_test, _y_test)
        (X_train, y_train), (X_test, y_test) = word2rep(_X_train, _y_train, self.model), word2rep(_X_test, _y_test,
                                                                                                      self.model)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        print("num of tests / num of labels: {} {} / {} {} in {:.2f} sec".format(
            len(X_train), len(X_test), len(set(y_train)), len(set(y_test)), time.time() - start_time))

        for c in SVM_Cs:
            clf = svm.SVC(C=c)
            if small_train:
                clf.fit(X_train[:SMALL_UCI_NUM], y_train[:SMALL_UCI_NUM])
            else:
                clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            with codecs.open(SOURCE_DIR + 'pred_UCI/debiased_' + config.MODEL_NAME +  str(c) + '_pred.txt', 'w', encoding='utf-8', errors='ignore') as f:
                for tokens, label in zip(_X_test, pred):
                    f.write('\t'.join(tokens) + '\t' + str(label) + '\n')
            print_result(y_test, pred, test_male_index, test_female_index)

        return 0

    def test_analogy(self):
        for w1, w2 in sensitive_pair:
            for word in neutral_word_list:
                try:
                    print('{}:{} = {}:{}'.format(
                        w1, w2, word, self.model.most_similar(positive=[w2, word], negative=[w1], topn=10)))
                except Exception as e:
                    continue



if __name__ == "__main__":
    print("corpus: {} / consider_gender: {}".format(config.MODEL_NAME, config.CONSIDER_GENDER))
    # 1. training w2v or preparing pre-trained embedding
    # python base_embedding.py or pass

    # 2. training transformed
    # python evaluate_methods.py
    # (paper) w2v_wiki_1108: sentiment/entity cutoff: 19.660907745361328/44.33755874633789, space_order=[-1, 1]
    # w2v_wiki_1123: sentiment/entity cutoff: -27.854839324951172/18.4766845703125, space_order=[1, -1]
    # w2v_redditsmall: sentiment/entity cutoff: 19.435272216796875/59.42977523803711, space_order=[1, -1]
    # (paper) w2v_reddit_1124: sentiment/entity cutoff: -25.70028305053711/19.568408966064453, space_order=[-1, -1]

    # 3. show statistics
    execute_models = ['w2v', 'my']
    uci_dataset = load_UCI()
    if 'w2v' in execute_models or 'deb' in execute_models:
        w2v = W2vModel(vocab_limit=100000)
        if 'w2v' in execute_models:
            w2v.test_analogy()
            #w2v.test_UCI(uci_dataset, small_train=True)
            #w2v.test_intrinsic()
        else:
            tmp_vectors = w2v.get_keyedvectors()
            w2v = {}
            deb = DebiasModel(deepcopy(tmp_vectors), same_env=True)
            deb.test_analogy()
            deb.test_UCI(uci_dataset, small_train=True)
            deb.test_intrinsic()
            deb = {}
        w2v = {}

    if 'my' in execute_models:
        # wiki (paper)
        if config.MODEL_NAME == 'wiki':
            my = MyModel(threshold=44.33755874633789, space_order=[-1, 1])
            my.test(uci_dataset, intensity_order=-1)
            my = MyModel(threshold=44.33755874633789, space_order=[-1, 1])
            my.test(uci_dataset, intensity_order=1)
        # reddit (paper)
        elif config.MODEL_NAME == 'reddit':
            my = MyModel(threshold=19.568408966064453, space_order=[-1, -1])
            my.test(uci_dataset, intensity_order=-1)
            my = MyModel(threshold=19.568408966064453, space_order=[-1, -1])
            my.test(uci_dataset, intensity_order=1)
        elif config.MODEL_NAME == 'glove':
            with codecs.open(config.MITIGATED_EMBEDDING_INFO, "r", encoding='utf-8', errors='ignore') as f:
                lines = re.split('[\r\n]+', f.read())
                sent_tokens = lines[0].rsplit(':', 2)
                entity_tokens = lines[1].rsplit(':', 2)
                threshold = float(entity_tokens[1].strip().split(' ')[0])
                space_order = [int(sent_tokens[2].strip()), int(entity_tokens[2].strip())]
                my = MyModel(threshold=threshold, space_order=space_order)
                my.test(uci_dataset, intensity_order=1)
                my = MyModel(threshold=threshold, space_order=space_order)
                my.test(uci_dataset, intensity_order=-1)

    print("end")
