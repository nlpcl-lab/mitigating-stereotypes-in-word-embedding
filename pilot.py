# -*- coding: utf-8 -*-
# Emotional_Word_Dictionary_RES_v1.2: 정규표현식 ^[a-zA-Z0-9]+\t[a-zA-Z0-9_]+\t[가-힣]+/[a-zA-Z]+\t 를 통해 오류줄 색인 가능.
# gender / sentiment / gender_pair words are filtered if it is oov, duplicated word, or words in both groups.
# e.g. '화나/A' in both a positive vocab and a negative vocab.
# w2v_model.wv.syn0norm is setting after most_similar() used or init_sims() used
# a = re.findall('\([^)]*\)',s) => ['(1,2,3,4,5)', '(5,4,3,2,1)']
# condition1 추가: 남, 녀는 동음이의어로 다른 뜻으로 쓰이게되는 경우 있어서 제외

import json, codecs, time, re, os
import config
import gensim
import math
import numpy as np
from gensim import utils, matutils
from gensim.models import word2vec, FastText
from collections import OrderedDict
from konlpy.tag import Twitter; t = Twitter()


COLLECTED_FNAME = config.COLLECTED_FNAME_TWITTER
COLLECTED_DATASET_DIR = 'source\\'
MODEL_NAME = 'twitter_all'
# MODEL_NAME = 'news2018'
DELTA_THRESHOLD = 1
GAMMA = 0.001   # Linguistic Regularities in Sparse and Explicit Word Representations
L_CUTOFF = 5 / 100
U_CUTOFF = 7.5 / 100

start_time = time.time()


def read_community_posting_and_sav_file():
    # fname -> json formatted file
    def read_file(fname):
        content = ''

        with codecs.open("../dataset/posting dataset/postings.json", "r", encoding="utf-8", errors='ignore') as f:
            content = json.load(f, object_pairs_hook=OrderedDict)

        return content

    # I still need this to take multiple datasets and assign a sheet to each.
    # for spss sav file read
    import savReaderWriter
    def showSavFiles():
        # print(read_file(''))
        savFileName = "../dataset/IAT dataset/Sexuality IAT.public.2017.sav"
        """
        reader_np = savReaderWriter.SavReaderNp("IAT dataset/Sexuality IAT.public.2017.sav")
        intermediate = reader_np.to_structured_array("Sexuality IAT.public.2017.dat")
        np.savetxt("Sexuality IAT.public.2017.csv", intermediate, delimiter=",")
        reader_np.close()
        """
        with savReaderWriter.SavReader(savFileName) as reader:
             for i, line in enumerate(reader):
                 if i >= 100:
                     exit()
                 print(line)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class EmbeddingTester(object):
    def __init__(self, is_selected_gender_vocab=False, remove_oov=True):
        """
        :param is_selected_gender_vocab: 'True' means selected_gender_vocab is prepared.
        :param remove_oov: remove words not in w2v.model vocab.
        """
        # embedding models
        self.w2v_fname = config.MODEL_DIR + 'w2v_{0}_sg_300_hs0_neg10_sampled_it10.model'.format(MODEL_NAME)
        self.fasttext_fname = config.MODEL_DIR + 'fasttext_{0}_sg_300_hs0_neg10_sampled_it10.model'.format(MODEL_NAME)
        self.w2v_model = self.load_w2v_model(self.w2v_fname)
        self.w2v_model.init_sims()                          # for using wv.syn0norm
        # self.fasttext_model = self.load_fasttext_model(self.fasttext_fname)

        # For in-out computation
        self.outv = gensim.models.KeyedVectors(vector_size=300)
        self.outv.vocab = self.w2v_model.wv.vocab
        self.outv.index2word = self.w2v_model.wv.index2word
        self.outv.syn0 = self.w2v_model.syn1neg

        # parameters
        self.l_cutoff = int(len(self.w2v_model.wv.index2word) * L_CUTOFF)
        self.u_cutoff = int(len(self.w2v_model.wv.index2word) * U_CUTOFF)

        # vocabs
        if is_selected_gender_vocab:
            self.gender_vocab = self.get_selected_gender_vocab(remove_oov=remove_oov)
        else:
            self.collect_gender_vocab(self.w2v_model)
        self.sentiment_vocab = self.get_sentiment_vocab(debug_mode=False, remove_oov=remove_oov)
        self.gender_pair_list = self.get_gender_pair_list(remove_oov=remove_oov)
        """
        gender_removed_vocab = {word: vocab_obj for word, vocab_obj in self.w2v_model.wv.vocab.items()
                                     if not (word in self.gender_vocab['0'] + self.gender_vocab['1'] or
                                             re.search(r'(/NP|/R|/n)$', word) or
                                             re.search(r'(남/N|녀/N|여/N|남자/N|여자/N|모/N|부/N|딸/N|아들/N|엄마/N|'
                                                       r'아빠/N|형/N|언니/N|오빠/N|누나/N|계집/N|공주/N|왕자/N|'
                                                       r'아버지/N|어머니/N|아내/N|어미/N|아비/N|아범/N|어멈/N|게이/N|'
                                                       r'레즈비언/N|년/N|놈/N)$', word) or
                                             re.search(r'^(남|녀|여|남자|여자|계집|공주|왕자|아버지|어머니|아내|어미|'
                                                       r'아비|아범|어멈|게이|레즈)', word))}
        """
        gender_removed_vocab = {word: vocab_obj for word, vocab_obj in self.w2v_model.wv.vocab.items()
                                if not (re.search(r'(/NP|/R|/n)$', word))}
        self.gender_removed_vocab = OrderedDict(sorted(gender_removed_vocab.items(), key=lambda item: -item[1].count))
        self.gender_neutral_vocab = self.collect_gender_neutral_vocab(setting=4)
        self.rep_idx = {word: i for i, word in enumerate(self.w2v_model.wv.index2word)}

    def _remove_oov(self, input_list):
        """
        :param input_list: list of words
        :return: oov removed list
        """
        tmp_list = []
        for word in input_list:
            if isinstance(word, tuple):
                tmp_tuple = ()
                for token in word:
                    try:
                        if isinstance(self.w2v_model[token], np.ndarray):
                            tmp_tuple += (token,)
                    except Exception as e:
                        continue
                tmp_list.append(tmp_tuple)
            else:
                try:
                    if isinstance(self.w2v_model[word], np.ndarray):
                        tmp_list.append(word)
                except Exception as e:
                    continue

        return tmp_list

    def _remove_duplicated_words(self, group1, group2):
        """
        :param group1: list of words in group1
        :param group2: list of words in group2
        :return: two list duplicated words are removed
        """
        group1 = set(group1)
        group2 = set(group2)
        intersection_group = group1.intersection(group2)
        group1 = list(group1 - intersection_group)
        group2 = list(group2 - intersection_group)
        return group1, group2

    def load_w2v_model(self, fname):
        try:
            print('Loading W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            w2v_model = word2vec.Word2Vec.load(fname)
            print(w2v_model)

        except IOError:
            print('No existed model. Training W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
            texts_ko = config.NewsCorpus(COLLECTED_FNAME)
            w2v_model = word2vec.Word2Vec(texts_ko, workers=4, hs=0, sg=1, size=300, window=5, min_count=5,
                                          sample=10 ^ -5, negative=10, alpha=0.025, min_alpha=0.0001, seed=1, iter=10)
            # init_sims: reduce memory but cannot continue training (because original vectors are removed.
            w2v_model.init_sims(replace=True)

            w2v_model.save(fname)  # save model

        print('Success to load W2v Model... in {0:.2f} seconds'.format(time.time() - start_time))
        return w2v_model

    def collect_gender_vocab(self, model):
        """
        Collect gender vocab based on seed subword, of which the number is min_count=5.
        :param model: word embedding model
        :return: nothing(make gender_vocab in the directory)
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_vocab_{0}.txt'.format(MODEL_NAME), "w", encoding='utf-8', errors='ignore') as write_file:
            gender_vocab = {}
            for word, vocab_obj in sorted(model.wv.vocab.items(), key=lambda item: -item[1].count):
                match = re.search(r'(남/N|녀/N|여/N|남자/N|여자/N|모/N|부/N|딸/N|아들/N|엄마/N|아빠/N|형/N|언니/N|'
                                  r'오빠/N|누나/N|계집/N|공주/N|왕자/N|아버지/N|어머니/N|아내/N|어미/N|아비/N|아범/N|'
                                  r'어멈/N|게이/N|레즈비언/N|년/N|놈/N)$', word)
                match = match or re.search(r'^(남|녀|여|남자|여자|계집|공주|왕자|아버지|어머니|아내|어미|아비'
                                           r'|아범|어멈|게이|레즈)', word)

                if match:
                    gender_vocab[word] = vocab_obj
                    write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            # print(sorted(gender_vocab.items(), key=lambda item: -item[1].count)[:10])
            print("Success to save gender vocabulary.")

    def get_selected_gender_vocab(self, remove_oov=True):
        """
        Return gender vocab(need the collected and 'selected' gender vocab in the directory)
        :return: gender vocab (dict - 0: list of words(woman), 1: list of words(man))
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_vocab_manuallyselected.txt'.format(MODEL_NAME), "r", encoding='utf-8',
                         errors='ignore') as read_file:
            # 1. Get gender vocabs.
            gender_vocab = OrderedDict()
            gender_vocab['0'], gender_vocab['1'] = [], []
            for line in read_file.read().splitlines():
                if len(line) > 1:
                    tokens = line.split('\t')
                    gender_vocab[tokens[2]].append(tokens[0])
            vocab_count = len(gender_vocab['0']) + len(gender_vocab['1'])

            # 2. Remove duplicated words and words in both groups.
            gender_vocab['0'], gender_vocab['1'] = self._remove_duplicated_words(gender_vocab['0'], gender_vocab['1'])

            # 3. Remove words not in w2v.model vocab.
            if remove_oov:
                gender_vocab['0'] = self._remove_oov(gender_vocab['0'])
                gender_vocab['1'] = self._remove_oov(gender_vocab['1'])

            vocab_without_count = len(gender_vocab['0']) + len(gender_vocab['1'])
            print("The number of gender_vocab words / without oov and duplications: {0} / {1}"
                  .format(vocab_count, vocab_without_count))

        return gender_vocab

    def get_sentiment_vocab(self, debug_mode=False, remove_oov=True):
        """
        :param debug_mode: print log or not
        :return: sentiment_vocab (dict keys - positive, negative, each of them contains list of words(sentiment))
        """
        def postag_simpler(token):
            """
            :param token: word/pos
            :return: list of word/simple pos
            """
            result_list = []
            pos_dict = {'n': 'N', 'nb': 'N', 'nc': 'N', 'ncs': 'N', 'nct': 'N', 'nq': 'N', 'nca': 'NAa', 'c': 'VN',
                        'p': 'VN', 'px': 'VN', 'pa': 'A', 'pad': 'A', 'pv': 'VA', 'a': 'Na'}
            try:
                [word, poses] = token.split('/')
            except ValueError as e:
                print(token)
            try:
                for pos in pos_dict[poses]:
                    result_list.append(word + '/' + pos)
            except Exception:
                print("error token in sentiment dataset: {0}".format(token))
                exit(1)

            return result_list

        with codecs.open('../dataset/sentiment dataset/Emotional_Word_Dictionary_RES_v1.2.txt'.format(MODEL_NAME), "r",
                         encoding='utf-8', errors='ignore') as read_file:
            # 1. Get sentiment vocabs.
            sentiment_vocab = OrderedDict()
            sentiment_vocab['positive'], sentiment_vocab['negative'] = [], []
            if debug_mode:
                pos_list = []
                sentiment_list = []
                for line in read_file.read().splitlines():
                    if len(line) > 0 and line[0] == 'S':
                        tokens = line.split('\t')
                        pos_list.append(tokens[2].split('/')[-1])
                        sentiment_list.append(tokens[7])
                        for simpler_token in postag_simpler(tokens[2]):
                            sentiment_vocab[tokens[7]].append(simpler_token)

                print("Check pos_list and sentiment_list", set(pos_list), set(sentiment_list))
            else:
                for line in read_file.read().splitlines():
                    if len(line) > 0 and line[0] == 'S':
                        tokens = line.split('\t')
                        for simpler_token in postag_simpler(tokens[2]):
                            sentiment_vocab[tokens[7]].append(simpler_token)
            vocab_count = len(sentiment_vocab['positive']) + len(sentiment_vocab['negative'])

            # 2. Remove duplicated words and words in both groups.
            sentiment_vocab['positive'], sentiment_vocab['negative'] = \
                self._remove_duplicated_words(sentiment_vocab['positive'], sentiment_vocab['negative'])

            # 3. Remove words not in w2v.model vocab.
            if remove_oov:
                sentiment_vocab['positive'] = self._remove_oov(sentiment_vocab['positive'])
                sentiment_vocab['negative'] = self._remove_oov(sentiment_vocab['negative'])

            vocab_without_count = len(sentiment_vocab['positive']) + len(sentiment_vocab['negative'])
            print("The number of sentiment_vocab words / without oov and duplications: {0} / {1}"
                  .format(vocab_count, vocab_without_count))

        return sentiment_vocab

    def get_gender_pair_list(self, remove_oov=True):
        """
        Return gender pair vocab
        :param remove_oov: remove words not in w2v.model vocab.
        :return: gender pair list
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_pair.txt'.format(MODEL_NAME), "r", encoding='utf-8',
                         errors='ignore') as read_file:
            # 1. Get gender pairs.
            gender_pair_list = []
            for line in read_file.read().splitlines():
                if len(line) > 1:
                    tokens = line.split('\t')
                    gender_pair_list.append((tokens[0], tokens[1]))
            vocab_count = len(gender_pair_list)

            # 2. Remove words not in w2v.model vocab.
            if remove_oov:
                gender_pair_list = self._remove_oov(gender_pair_list)

            vocab_without_count = len(gender_pair_list)
            print("The number of gender_pairs / without oov and duplications: {0} / {1}"
                  .format(vocab_count, vocab_without_count))

        return gender_pair_list

    def load_fasttext_model(self, fname):
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
            print('No existed model. Training Fasttext Model... in {0:.2f} seconds'.format(time.time() - start_time))
            texts_ko = config.NewsCorpus(COLLECTED_FNAME)
            fasttext_model = FastText(texts_ko, workers=4, hs=0, sg=1, size=300, window=5, min_count=5,
                                 sample=10 ^ -5, negative=10, alpha=0.025, min_alpha=0.0001, seed=1, iter=10,
                                 min_n=2, max_n=3)
            fasttext_model.save(fname)

        print('Success to load Fasttext Model... in {0:.2f} seconds'.format(time.time() - start_time))
        return fasttext_model

    def make_test_analogy(self):
        """
        S(x,y) = three methods if x-y< = 1 else 0
        """
        def delta_threshold(x, y):
            return True if np.linalg.norm(self.w2v_model[x] - self.w2v_model[y]) <= DELTA_THRESHOLD else False

        def _cal_argmax_y(w2v_model, vocab_size, sort_index, elem123, count_threshold):
            y, y_score = '', 0
            count = 0  # the number of search trial with delta threshold
            for i in range(vocab_size[0]):
                y = w2v_model.wv.index2word[sort_index[i]]
                if x == y or a == x or b == y or a == y: #or y not in self.gender_removed_vocab:
                    continue
                elif not delta_threshold(x, y) and count < count_threshold:
                    count += 1
                    continue
                elif not delta_threshold(x, y) and count >= count_threshold:
                    y_score = 0
                    break
                else:
                    y_score = elem123[sort_index[i]]
                    break

            return y, y_score, count

        def _cal_cosmul(w2v_model, a, b, x, cos_yb, cos_yx, cos_ya, delta_list, count_threshold=1000):
            """
            :return:
            y = the word
            y_score = A score of the word
            count: the number of search trial with delta threshold
            """
            vocab_size = np.shape(cos_yb)
            elem1 = np.add(np.full(vocab_size, 1), cos_yb)
            elem2 = np.add(np.full(vocab_size, 1), cos_yx)
            elem3 = np.add(np.full(vocab_size, 1 + GAMMA), cos_ya)
            elem123 = np.true_divide(np.multiply(elem1, elem2), elem3)
            sort_index = np.argsort(-elem123)

            cond = (delta_list[sort_index] <= 1) & (np.array(w2v_model.wv.index2word)[sort_index] != a) & \
                   (np.array(w2v_model.wv.index2word)[sort_index] != b) & \
                   (np.array(w2v_model.wv.index2word)[sort_index] != x)
            y_list = np.array(w2v_model.wv.index2word)[sort_index][cond]
            y_score_list = elem123[sort_index][cond]

            for y, y_score in zip(y_list, y_score_list):
                if not re.search(r'(/NP|/R|/n)$', y_list):
                    return y, y_score, 0

            y = y_list[0] if len(y_list) > 0 else 'None'
            y_score = y_score_list[0] if len(y_score_list) > 0 else 0
            count = 0

            #y, y_score, count = _cal_argmax_y(w2v_model, vocab_size, sort_index, elem123, count_threshold)

            #print('COSMUL a b x y y_score {} {} {} {} {} delta {} {}'.format(a, b, x, y, y_score, delta_threshold(x, y),
            #                                                                 count))
            return y, y_score, count

        def _cal_cosadd(w2v_model, a, b, x, cos_yb, cos_yx, cos_ya, delta_list, count_threshold=1000):
            vocab_size = np.shape(cos_yb)
            elem123 = np.add(cos_yb, cos_yx) - cos_ya
            sort_index = np.argsort(-elem123)
            y, y_score, count = _cal_argmax_y(w2v_model, vocab_size, sort_index, elem123, count_threshold)

            #print('COSADD a b x y y_score {} {} {} {} {} delta {} {}'.format(a, b, x, y, y_score, delta_threshold(x, y),
            #                                                                 count))
            return y, y_score, count

        def _cal_pair(w2v_model, a, b, x, count_threshold=1000):
            vocab_size = (len(w2v_model.wv.index2word),)
            # np.dot(nd-array, 1d-array) => 1d-array (y_scores of all vocabs)
            elem1 = w2v_model[x] - w2v_model.wv.syn0norm
            elem2 = w2v_model[a] - w2v_model[b]
            elem3 = np.linalg.norm(elem1, axis=1)
            elem123 = np.dot(elem1, elem2) / (elem3 * (np.linalg.norm(elem2)))
            sort_index = np.argsort(-elem123)
            y, y_score, count = _cal_argmax_y(w2v_model, vocab_size, sort_index, elem123, count_threshold)

            print('PAIR a b x y y_score {} {} {} {} {} delta {} {}'.format(a, b, x, y, y_score, delta_threshold(x, y),
                                                                           count))
            return y, y_score, count

        def _cal_pair_compressed(w2v_model, a, b, x_list, x_index_list, count_threshold=1000):
            vocab_size = (len(w2v_model.wv.index2word),)
            x_size = len(x_index_list)
            elem1 = w2v_model.wv.syn0norm[x_index_list, :][:, None] - w2v_model.wv.syn0norm # memory error
            elem2 = w2v_model[a] - w2v_model[b]
            elem3 = np.linalg.norm(elem1, axis=2)
            elem123 = np.inner(elem1, elem2) / (elem3 * (np.linalg.norm(elem2)))            # y_score matrix
            sort_index = np.argsort(-elem123, axis=1)

            # 여기 잘못된 부분. elem3 적용하면 안되고 sort_index filter해서해야함
            boolean_delta_of_elem123 = (elem3 <= 1) & \
                                       (np.array(w2v_model.wv.index2word)[sort_index] != a) & \
                                       (np.array(w2v_model.wv.index2word)[sort_index] != b) & \
                                       (np.array(w2v_model.wv.index2word)[sort_index] !=
                                        np.tile(x_list.reshape(x_size, 1), vocab_size)) & \
                                       ~(re.search(r'(/NP|/R|/n)$', np.array(w2v_model.wv.index2word)[sort_index]))

            y_indexes, y_scores, counts = [], [], []
            for i, x in enumerate(x_index_list):
                y_indexes_cand = sort_index[i, :][boolean_delta_of_elem123[i, :]]
                y_scores_cand = elem123[i, :][boolean_delta_of_elem123[i, :]]

                y_indexes.append(y_indexes_cand[0]) if len(y_indexes_cand) > 0 else y_indexes.append(0)
                y_scores.append(y_scores_cand[0]) if len(y_scores_cand) > 0 else y_scores.append(0)

            counts = np.full(np.shape(y_indexes), 0)

            return list(zip(np.array(w2v_model.wv.index2word)[y_indexes], y_scores, counts))

        def calculate_cosine_scores(w2v_model, a, b, x, cos_ya, cos_yb, cos_yx):
            """
            Given gender pair (a,b), generate (x,y) pair which satisfies within delta threshold in descending order.
            :param w2v_model:
            :param man_word:
            :param woman_word:
            :return:
            """
            count_threshold = 1000
            delta_list = np.linalg.norm(w2v_model[x] - w2v_model.wv.syn0norm, axis=1)

            # 3COSMUL: (1 + cos_yb)(1 + cos_yx)) / (1 + cos_ya + GAMMA)
            mul_score_tuple = _cal_cosmul(w2v_model, a, b, x, cos_yb, cos_yx, cos_ya, delta_list, count_threshold=count_threshold)
            #mul_score_tuple = ('예시/N', 0, 0)
            # 3COSADD: cos_yb + cos_yx - cos_ya
            add_score_tuple = _cal_cosadd(w2v_model, a, b, x, cos_yb, cos_yx, cos_ya, delta_list, count_threshold=count_threshold)
            #add_score_tuple = ('예시/N', 0, 0)
            # PAIR: cos(a - b, x - y)
            # pair_score_tuple = _cal_pair(w2v_model, a, b, x, cos_yb, cos_yx, cos_ya, count_threshold=count_threshold)
            #pair_score_tuple = ('예시/N', 0, 0)

            return mul_score_tuple, add_score_tuple#, pair_score_tuple

        with codecs.open(COLLECTED_DATASET_DIR + 'gender_analogy_{0}.txt'.format(MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            analogy_pair_score_dict = {}
            x_list = list(set(list(self.gender_removed_vocab.keys())[25000:50000])) #- set(self.gender_vocab['0'] + self.gender_vocab['1']))
            x_index_list = [self.rep_idx[x] for x in x_list]
            #cos_yx_list = np.einsum('jk,ik->ij', self.w2v_model.wv.syn0norm, self.w2v_model.wv.syn0norm[x_index_list])
            # memory caution
            for (a, b) in self.gender_pair_list[:5]:
                if a == '남/N':
                    continue
                write_file.write('a\tb\tx\tmul\tadd\tpair\n')
                cos_ya = np.dot(self.w2v_model.wv.syn0norm, self.w2v_model[a])
                cos_yb = np.dot(self.w2v_model.wv.syn0norm, self.w2v_model[b])
                #pair_tuple_list = _cal_pair_compressed(self.w2v_model, a, b, x_list, x_index_list, count_threshold=1000)

                #for i, (x, pair_tuple) in enumerate(zip(x_list, pair_tuple_list)):
                for i, x in enumerate(x_list):
                    if i % int(len(x_list)/100 - 1) == 0:
                        print("{:.1f}% of neutral words have done with <{}, {}>".format(i * 100 / len(x_list), a, b))

                    cos_yx = np.inner(self.w2v_model.wv.syn0norm, self.w2v_model[x])
                    #mul_tuple, add_tuple, pair_tuple = calculate_cosine_scores(self.w2v_model, a, b, x)
                    mul_tuple, add_tuple = calculate_cosine_scores(self.w2v_model, a, b, x, cos_ya, cos_yb, cos_yx)
                    pair_tuple = ('예시/N', 0, 0)

                    # Given x, if delta_threshold > 1 for all words, y cannot be maken and y_score is 0. 
                    
                    if mul_tuple[2] > 0 or add_tuple[2] > 0 or pair_tuple[2] > 0:
                        analogy_pair_score_dict[(a, b, x)] = (mul_tuple, add_tuple, pair_tuple)
                        write_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(a, b, x, mul_tuple, add_tuple, pair_tuple))

                # items(): item[a][b][c], a:0 or 1(key or value) b: 0~2 (tuple th) c: 0~2(y, y_score, count)
                write_file.write('top 150 list - mul\n')
                for (a, b, x), (mul_tuple, add_tuple, pair_tuple) in sorted(analogy_pair_score_dict.items(),
                                                                            key=lambda item: -item[1][0][1])[:150]:
                    write_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(a, b, x, mul_tuple, add_tuple, pair_tuple))
                write_file.write('top 150 list - add\n')
                for (a, b, x), (mul_tuple, add_tuple, pair_tuple) in sorted(analogy_pair_score_dict.items(),
                                                                            key=lambda item: -item[1][1][1])[:150]:
                    write_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(a, b, x, mul_tuple, add_tuple, pair_tuple))
                """
                write_file.write('top 150 list - pair\n')
                for (a, b, x), (mul_tuple, add_tuple, pair_tuple) in sorted(analogy_pair_score_dict.items(),
                                                                            key=lambda item: -item[1][2][1])[:150]:
                    write_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(a, b, x, mul_tuple, add_tuple, pair_tuple))
                """

        return 0

    def _gender_neutral_definition_1(self):
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_neutral_{0}.txt'.format(MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            gender_neutral_vocab = OrderedDict()
            for word, vocab_obj in sorted(self.w2v_model.wv.vocab.items(), key=lambda item: -item[1].count):
                match = re.search(r'(/A|/a|/V)$', word)
                if match:
                    gender_neutral_vocab[word] = vocab_obj
                    write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            print("Success to save gender_neutral vocabulary.")

        return gender_neutral_vocab

    def _gender_neutral_definition_2(self):
        """
        Setting 2: just remove the gender_specific_vocab of news.
        :return:
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_neutral_{0}_2.txt'.format(MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            gender_neutral_vocab = OrderedDict()
            w2v_vocab = {word: vocab_obj for word, vocab_obj in self.w2v_model.wv.vocab.items()
                         if word not in self.gender_vocab['0'] + self.gender_vocab['1']}
            for word, vocab_obj in sorted(w2v_vocab.items(), key=lambda item: -item[1].count):
                gender_neutral_vocab[word] = vocab_obj
                write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            print("Success to save gender_neutral_2 vocabulary.")

        return gender_neutral_vocab

    def _gender_neutral_definition_3(self):
        """
        Setting 3: remove noun particle / foreign words / digit and gender_specific suffix / prefix.
                After that, choose the number of top 5% from the sorted vocab by count*rank.
        :return:
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_neutral_{0}_3.txt'.format(MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            gender_neutral_vocab = OrderedDict()
            tmp_list = []
            sorted_w2v_list = sorted(self.gender_removed_vocab.items(), key=lambda item: -item[1].count)
            for i, (word, vocab_obj) in enumerate(sorted_w2v_list):
                # key=lambda item: -(item[1].count * item[0]):
                tmp_list.append((word, vocab_obj, i))

            for word, vocab_obj, i in sorted(tmp_list, key=lambda item: -(item[1].count * item[2]))[:int(len(self.w2v_model.wv.index2word) * 5/100)]:
                gender_neutral_vocab[word] = vocab_obj
                write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            print("Success to save gender_neutral_3 vocabulary.")

        return gender_neutral_vocab

    def _gender_neutral_definition_4(self):
        """
        Setting 4: remove noun particle / foreign words / digit and gender_specific suffix / prefix.
                After that, only remain the data between upper and lower cut off based on frequency.
        :return:
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_neutral_{0}_4.txt'.format(MODEL_NAME), "w", encoding='utf-8',
                         errors='ignore') as write_file:
            gender_neutral_vocab = OrderedDict()
            tmp_list = []
            for word, vocab_obj in sorted(self.gender_removed_vocab.items(), key=lambda item: -item[1].count)[self.l_cutoff:self.u_cutoff]:
                gender_neutral_vocab[word] = vocab_obj
                write_file.write('{0}\t{1}\n'.format(word, vocab_obj.count))

            print("Success to save gender_neutral_4 vocabulary.")

        return gender_neutral_vocab

    def collect_gender_neutral_vocab(self, setting=1):
        self.case_name = "_gender_neutral_definition_" + str(setting)
        gender_neutral_vocab = getattr(self, self.case_name, lambda: "1")()
        print("The number of gender_neutral words / without oov and duplications: {0} / same"
              .format(len(gender_neutral_vocab.items())))
        return gender_neutral_vocab

    # ### bias calculation zone ### #

    def _cal_cosine_inout(self, word1, word2):
        """
        The method is cosine similarity with in-out vectors.
        Some of In-out similarity have negative. Generally, it has small number.
        note.
        self.w2v_model[word1]: shape 300 / unit vector
        self.outv[word2]: shape 300 / not unit vector
        cosine sim: np.dot(self.w2v_model[word1], self.outv[word2]) /
            (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.outv[word2]))
        ### not use ### self.w2v_model.wv.word_vec(word1, use_norm=True): same
        ### not use ### self.outv.wv.word_vec(word2, use_norm=True): shape 300 / unit vector
        ### not use ### np.dot(self.w2v_model.wv.word_vec(word1, use_norm=True), self.outv.wv.word_vec(word2, use_norm=True))
        :return:
        """
        return np.dot(self.w2v_model[word1], self.outv[word2]) / \
               (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.outv[word2]))

    def _cal_cosine_sigmoid_inout(self, word1, word2):
        return sigmoid(np.dot(self.w2v_model[word1], self.outv[word2]) / \
               (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.outv[word2])))

    def _cal_cosine_inin(self, word1, word2):
        """
        Default method is cosine similarity with in-in vectors.
        same codes.
        self.w2v_model.similarity(word1, word2)
        np.dot(matutils.unitvec(self.w2v_model[word1]), matutils.unitvec(self.w2v_model[word2]))
        np.dot(self.w2v_model[word1], self.w2v_model[word2]) /\
               (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.w2v_model[word2]))
        :return:
        """
        return self.w2v_model.similarity(word1, word2)

    def _cal_relative_dist(self, word1, word2):
        """
        The smaller relative_dist is, the closer that sentiment of that gender has.
        :param word1:
        :param word2:
        :return:
        """
        return -np.linalg.norm(self.w2v_model[word1] - self.w2v_model[word2])

    def _cal_setting1(self, man_words, pos_words, neg_words, similarity_method, gender='man'):
        total_pos_score, total_neg_score = 0, 0
        self.case_name = "_cal_" + similarity_method
        for word1 in man_words:
            pos_score, neg_score = 0, 0
            for word2 in pos_words:
                pos_score += getattr(self, self.case_name, lambda: "cosine_inin")(word1, word2)
            pos_score = pos_score / len(pos_words)
            total_pos_score += pos_score

            for word2 in neg_words:
                neg_score += getattr(self, self.case_name, lambda: "cosine_inin")(word1, word2)
            neg_score = neg_score / len(neg_words)
            total_neg_score += neg_score
            # print('total_pos_neg_score', total_pos_score, total_neg_score)

        total_pos_score = total_pos_score / len(man_words)
        total_neg_score = total_neg_score / len(man_words)
        print('{0} words are {1} ({2:.3f} {3} {4:.3f}).'.format(gender,
              'positive' if total_pos_score > total_neg_score else 'negative', total_pos_score,
              '>' if total_pos_score > total_neg_score else '<', total_neg_score))
        result_score = self._softmax_score(total_pos_score, total_neg_score)
        print('the softmax score (e(pos)/e(pos)+e(neg)): {0:.3f}'.format(result_score))
        return result_score

    def _cal_setting2(self, man_word, woman_word, pos_words, similarity_method, sentiment='positive', debug_mode=False):
        man_score = 0
        self.case_name = "_cal_" + similarity_method
        for j, pos_word in enumerate(pos_words):
            man_dist = getattr(self, self.case_name, lambda: "cosine_inin")(man_word, pos_word)
            woman_dist = getattr(self, self.case_name, lambda: "cosine_inin")(woman_word, pos_word)
            if debug_mode and j < 10:
                print('Given {0}: {1} {2:.3f} {3} {4:.3f}'.format(pos_word, man_word, man_dist, woman_word, woman_dist))
            man_score = man_score + (man_dist - woman_dist)
        # print(man_word, woman_word, '~ {0} word'.format(sentiment), man_score)
        return man_score

    def _softmax_score(self, *args):
        return (np.exp(args) / np.exp(args).sum(axis=0))[0]

    def cal_sentiment_bias(self, similarity_method, group=2, setting=2, debug_mode=False):
        """
        similarity_method: bias are measured between two words.
        :return:
        """
        if group == 1:
            man_words, woman_words = self.gender_vocab['1'], self.gender_vocab['0']
        elif group == 2:
            man_words, woman_words = zip(*self.gender_pair_list)

        pos_words, neg_words = self.sentiment_vocab['positive'], self.sentiment_vocab['negative']

        if setting == 1:
            man_score = self._cal_setting1(man_words, pos_words, neg_words, similarity_method, gender='man')
            woman_score = self._cal_setting1(woman_words, pos_words, neg_words, similarity_method, gender='woman')
            print('{0} ({1:.6f} {2} {3:.6f})'.format('man positive' if man_score > woman_score else 'woman positive',
                                                     man_score, '>' if man_score > woman_score else '<', woman_score))

        elif group == 2 and setting == 2:
            man_score, total_score = 0, 0
            for i, (man_word, woman_word) in enumerate(self.gender_pair_list):
                man_score = self._cal_setting2(man_word, woman_word, pos_words, similarity_method, sentiment='positive')
                man_score -= self._cal_setting2(man_word, woman_word, neg_words, similarity_method, sentiment='negative')
                total_score = total_score + (man_score / len(pos_words + neg_words))
            total_score = total_score / len(self.gender_pair_list)
            print('{0} positive ({1:.6f})'.format('man' if total_score > 0 else 'woman', total_score))

        print("[{0}] sentiment bias is calculated.\n".format(similarity_method))
        return 0

    def cal_gender_bias(self, similarity_method, debug_mode=False):
        gender_neutral_words = self.gender_neutral_vocab.keys()
        stereotype_list = []
        self.case_name = "_cal_" + similarity_method

        for neutral_word in gender_neutral_words:
            stereotype_list.append((neutral_word,
                                      sum([getattr(self, self.case_name, lambda: "cosine_inin")(man_word, neutral_word)
                                          - getattr(self, self.case_name, lambda: "cosine_inin")(woman_word,
                                                                                                 neutral_word)
                                          for (man_word, woman_word) in self.gender_pair_list])))

        _, stereotype_scores = zip(*stereotype_list)
        print("top difference (man): {0}".format(sorted(stereotype_list, key=lambda item: -item[1])[:10]))
        print("top difference (woman): {0}".format(sorted(stereotype_list, key=lambda item: item[1])[:10]))
        total_score_avg, total_score_std = np.average(stereotype_scores), np.std(stereotype_scores)
        print('stereotype score ({0}): lean to {1} ({2:.6f}) / std {3:.6f}'
              .format(similarity_method, 'man' if total_score_avg > 0 else 'woman', total_score_avg, total_score_std))

        return 0

    # ### bias calculation zone end ### #

    def pliot_similarity_test(self):
        """
        Test for similarity
        :return:
        """
        word1 = '남자/N'
        word2 = '여자/N'
        neu1 = '피의자/N'
        neu2 = '행복/N'
        print(neu1, self.w2v_model.similarity(word1, neu1), self.w2v_model.similarity(word2, neu1))
        print(neu2, self.w2v_model.similarity(word1, neu2), self.w2v_model.similarity(word2, neu2))
        print(word1, self.w2v_model.most_similar(word1))
        print(word2, self.w2v_model.most_similar(word2))
        print('neu1+word1-word2', self.w2v_model.most_similar([word1, neu1], negative=[word2]))
        print('neu2+word1-word2', self.w2v_model.most_similar([word1, neu2], negative=[word2]))
        print('neu1-word1+word2', self.w2v_model.most_similar([word2, neu1], negative=[word1]))
        print('neu2-word1+word2', self.w2v_model.most_similar([word2, neu2], negative=[word1]))
        print(neu1, self.w2v_model.most_similar(neu1))
        print(neu2, self.w2v_model.most_similar(neu2))
        print(neu1, self.fasttext_model.similarity(word1, neu1), self.fasttext_model.similarity(word2, neu1))
        print(neu2, self.fasttext_model.similarity(word1, neu2), self.fasttext_model.similarity(word2, neu2))
        print(word1, self.fasttext_model.most_similar(word1))
        print(word2, self.fasttext_model.most_similar(word2))
        print('neu1+word1-word2', neu1, self.fasttext_model.most_similar([word1, neu1], negative=[word2]))
        print('neu2+word1-word2', neu2, self.fasttext_model.most_similar([word1, neu2], negative=[word2]))
        print(neu1, self.fasttext_model.most_similar(neu1))
        print(neu2, self.fasttext_model.most_similar(neu2))
        print('---------------------------------------------------')
        word1 = '남자/N'
        word2 = '여자/N'
        neu1 = '게임/N'
        neu2 = '화장품/N'
        print(neu1, self.w2v_model.similarity(word1, neu1), self.w2v_model.similarity(word2, neu1))
        print(neu2, self.w2v_model.similarity(word1, neu2), self.w2v_model.similarity(word2, neu2))
        print(word1, self.w2v_model.most_similar(word1))
        print(word2, self.w2v_model.most_similar(word2))
        print('neu1+word1-word2', self.w2v_model.most_similar([word1, neu1], negative=[word2]))
        print('neu2+word1-word2', self.w2v_model.most_similar([word1, neu2], negative=[word2]))
        print('neu1-word1+word2', self.w2v_model.most_similar([word2, neu1], negative=[word1]))
        print('neu2-word1+word2', self.w2v_model.most_similar([word2, neu2], negative=[word1]))
        print(neu1, self.w2v_model.most_similar(neu1))
        print(neu2, self.w2v_model.most_similar(neu2))
        print(neu1, self.fasttext_model.similarity(word1, neu1), self.fasttext_model.similarity(word2, neu1))
        print(neu2, self.fasttext_model.similarity(word1, neu2), self.fasttext_model.similarity(word2, neu2))
        print(word1, self.fasttext_model.most_similar(word1))
        print(word2, self.fasttext_model.most_similar(word2))
        print('neu1+word1-word2', neu1, self.fasttext_model.most_similar([word1, neu1], negative=[word2]))
        print('neu2+word1-word2', neu2, self.fasttext_model.most_similar([word1, neu2], negative=[word2]))
        print(neu1, self.fasttext_model.most_similar(neu1))
        print(neu2, self.fasttext_model.most_similar(neu2))

    def sent_bias_test(self):
        self.cal_sentiment_bias(similarity_method='cosine_inout')
        self.cal_sentiment_bias(similarity_method='cosine_inin')
        self.cal_sentiment_bias(similarity_method='cosine_sigmoid_inout')
        self.cal_sentiment_bias(similarity_method='relative_dist')

    def gender_bias_test(self):
        self.cal_gender_bias(similarity_method='cosine_inout')
        self.cal_gender_bias(similarity_method='cosine_inin')
        self.cal_gender_bias(similarity_method='cosine_sigmoid_inout')
        self.cal_gender_bias(similarity_method='relative_dist')


if __name__ == '__main__':
    # First, is_selected_gender_vocab=False
    # collect_gender_vocab(w2v_model)
    # Second,
    # after manually selecting gender_vocab with changing file name 'gender_vocab_manuallyselected.txt'
    # do is_selected_gender_vocab=True
    et = EmbeddingTester(is_selected_gender_vocab=True, remove_oov=True)
    # et.sent_bias_test()
    # et.gender_bias_test()
    et.make_test_analogy()
    # et.prior_similarity_test()

