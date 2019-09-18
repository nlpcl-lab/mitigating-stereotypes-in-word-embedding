"""
Revised by Huije Lee
Code reference
William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from
Unlabeled Corpora. Proceedings of EMNLP. 2016. (to appear; arXiv:1606.02820).
"""
import heapq
import _pickle as cPickle
import codecs
import re
import numpy as np
import config
import gensim

import config
from collections import OrderedDict
from gensim.test.utils import datapath
from gensim.models import word2vec
from sklearn import preprocessing


DEFAULT_ARGUMENTS_W2V = dict(workers=4, sg=1, size=300, window=5, min_count=5, sample=10^-4, negative=5, seed=1, iter=2)
sensitive_pair, neutral_word_list = config.load_analogy_pair(config.SOURCE_DIR + 'minority_groups.txt')


def load_pickle(fname):
    with open(fname) as f:
        return cPickle.load(f)


def lines(fname):
    with codecs.open(fname, "r", encoding="utf-8", errors='ignore') as f:
        for line in f:
            yield line


class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = {w: i for i, w in enumerate(self.iw)}
        if normalize:
            self.normalize()

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    @classmethod
    def load(cls, path, normalize=True, add_context=True, **kwargs):
        mat = np.load(path + "-w.npy")
        if add_context:
            mat += np.load(path + "-c.npy")
        iw = load_pickle(path + "-vocab.pkl")
        return cls(mat, iw, normalize)

    def get_subembed(self, word_list, **kwargs):
        word_list = [word for word in word_list if not self.oov(word)]
        keep_indices = [self.wi[word] for word in word_list]
        return Embedding(self.m[keep_indices, :], word_list, normalize=False)

    def reindex(self, word_list, **kwargs):
        new_mat = np.empty((len(word_list), self.m.shape[1]))
        valid_words = set(self.iw)
        for i, word in enumerate(word_list):
            if word in valid_words:
                new_mat[i, :] = self.represent(word)
            else:
                new_mat[i, :] = 0
        return Embedding(new_mat, word_list, normalize=False)

    def get_neighbourhood_embed(self, w, n=1000):
        neighbours = self.closest(w, n=n)
        keep_indices = [self.wi[neighbour] for _, neighbour in neighbours]
        new_mat = self.m[keep_indices, :]
        return Embedding(new_mat, [neighbour for _, neighbour in neighbours])

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def oov(self, w):
        return not (w in self.wi)

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            print("OOV: ", w)
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))


class WordEmbedding(Embedding):
    def __init__(self, path, words, dim=300, normalize=True, **kwargs):
        seen = []
        vs = {}
        for line in lines(path):
            split = line.split()
            w = split[0]
            if w in words:
                seen.append(w)
                vs[w] = np.array(list(map(float, split[1:])), dtype='float32')
        self.iw = seen
        self.wi = {w: i for i, w in enumerate(self.iw)}
        self.m = np.vstack(vs[w] for w in self.iw)
        if normalize:
            self.normalize()


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
                print('Loading W2v Model... in {0:.2f} seconds'.format(config.whattime()))
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
                    print('Success to load W2v Model... in {0:.2f} seconds'.format(config.whattime()))
                    return tmp_w2v

                print(w2v_model)
                print('Success to load W2v Model... in {0:.2f} seconds'.format(config.whattime()))
                return w2v_model

        except Exception as e:
            print('No existed model. Training W2v Model... in {0:.2f} seconds'.format(config.whattime()))
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

        print('Success to load W2v Model... in {0:.2f} seconds'.format(config.whattime()))

        return w2v_model.wv

    def test_intrinsic(self):
        try:
            self.w2v_model.wv.accuracy(config.SOURCE_DIR+'questions-words.txt', restrict_vocab=300000)
            """
            analogy_score, result_list = self.w2v_model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
            print("score: {:.2f}".format(analogy_score))
            for result_dict in result_list:
                print("{}: True {} / False {}".format(result_dict['section'], result_dict['correct'][:3], result_dict['incorrect'][:3]))
            """
        except Exception as e:
            self.w2v_model.accuracy(config.SOURCE_DIR + 'questions-words.txt', restrict_vocab=300000)
        try:
            similarities = self.w2v_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)
        except Exception as e:
            similarities = self.w2v_model.evaluate_word_pairs(datapath('wordsim353.tsv'), restrict_vocab=300000)

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
        with codecs.open(config.SOURCE_DIR + '{}_vocabs.txt'.format(config.MODEL_NAME), "w", encoding='utf-8',
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

