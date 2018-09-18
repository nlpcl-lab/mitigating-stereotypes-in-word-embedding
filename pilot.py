# -*- coding: utf-8 -*-
# Emotional_Word_Dictionary_RES_v1.2: 정규표현식 ^[a-zA-Z0-9]+\t[a-zA-Z0-9_]+\t[가-힣]+/[a-zA-Z]+\t 를 통해 오류줄 색인 가능.
# gender / sentiment / gender_pair words are filtered if it is oov, duplicated word, or words in both groups.
# e.g. '화나/A' in both a positive vocab and a negative vocab.

import json, codecs, time, re
import config
import gensim
import math
import numpy as np
from gensim import utils, matutils
from gensim.models import word2vec, FastText
from collections import OrderedDict
from konlpy.tag import Twitter; t = Twitter()


COLLECTED_FNAME = config.COLLECTED_FNAME_NEWS
COLLECTED_DATASET_DIR = 'source\\'
MODEL_NAME = 'twitter_all'
# MODEL_NAME = 'news2018'

start_time = time.time()


def read_community_posting_and_Sav_File():
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
        self.w2v_fname = config.MODEL_DIR + 'w2v_{0}_sg_300_hs0_neg10_sampled_it10.model'.format(MODEL_NAME)
        self.fasttext_fname = config.MODEL_DIR + 'fasttext_{0}_sg_300_hs0_neg10_sampled_it10.model'.format(MODEL_NAME)
        self.w2v_model = self.load_w2v_model(self.w2v_fname)
        # self.fasttext_model = self.load_fasttext_model(self.fasttext_fname)

        # For in-out computation
        self.outv = gensim.models.KeyedVectors(vector_size=300)
        self.outv.vocab = self.w2v_model.wv.vocab
        self.outv.index2word = self.w2v_model.wv.index2word
        self.outv.syn0 = self.w2v_model.syn1neg

        if is_selected_gender_vocab:
            self.gender_vocab = self.get_selected_gender_vocab(remove_oov=remove_oov)
        else:
            self.collect_gender_vocab(self.w2v_model)
        self.sentiment_vocab = self.get_sentiment_vocab(debug_mode=False, remove_oov=remove_oov)
        self.gender_pair_list = self.get_gender_pair_list(remove_oov=remove_oov)

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
        :return: only print
        """
        self.w2v_model.wv.vocab

    def _cal_cosine_inout(self, word1, word2):
        """
        The method is cosine similarity with in-out vectors.
        Some of In-out similarity have negative. Generally, it has small number.
        note.
        self.w2v_model[word1]: shape 300 / unit vector
        self.outv[word2]: shape 300 / not unit vector
        cosine sim: np.dot(self.w2v_model[word1], self.outv[word2]) /
            (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.outv[word2])))
        ### not use ### self.w2v_model.wv.word_vec(word1, use_norm=True): same
        ### not use ### self.outv.wv.word_vec(word2, use_norm=True): shape 300 / unit vector
        ### not use ### np.dot(self.w2v_model.wv.word_vec(word1, use_norm=True), self.outv.wv.word_vec(word2, use_norm=True))
        :return:
        """
        print(word1, word2)
        print(np.dot(self.w2v_model[word1], self.outv[word2]) /\
               (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.outv[word2])))

        return np.dot(self.w2v_model[word1], self.outv[word2]) / \
               (np.linalg.norm(self.w2v_model[word1]) * np.linalg.norm(self.outv[word2]))

    def _cal_default(self, word1, word2):
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
        return self.w2v_model[word1] - self.w2v_model[word2]



    def cal_sentiment_bias(self, similarity_method='cosine_inout'):
        """

        :return:
        """
        self.case_name = "_cal_" + similarity_method
        man_words, woman_words = self.gender_vocab['0'], self.gender_vocab['1']
        pos_words, neg_words = self.sentiment_vocab['positive'], self.sentiment_vocab['negative']
        pos_score = 0
        neg_score = 0
        for word1 in man_words:
            for word2 in pos_words:
                pos_score += getattr(self, self.case_name, lambda: "default")(word1, word2)
                print(pos_score)

        return 1

    def similarity_test(self):
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

    def definition_test(self):
        # Test - 성 임베딩 차원 규명 47 pair
        gender_diff_vec_list = []
        sentiment_invocab_list = []
        """
        for (word1, word2) in self.gender_pair_list:
            print(word1, word2, self.w2v_model.most_similar([word1], negative=[word2]))
            gender_diff_vec_list.append(self.w2v_model[word1] - self.w2v_model[word2])

        for word in self.sentiment_vocab['positive'] + self.sentiment_vocab['negative']:
            if word in self.w2v_model.wv.vocab:
                sentiment_invocab_list.append(word)
            else:
                print(word)
        """

        self.cal_sentiment_bias()


if __name__ == '__main__':
    # First, is_selected_gender_vocab=False
    # collect_gender_vocab(w2v_model)
    # Second,
    # after manually selecting gender_vocab with changing file name 'gender_vocab_manuallyselected.txt'
    # do is_selected_gender_vocab=True
    et = EmbeddingTester(is_selected_gender_vocab=True, remove_oov=True)
    et.definition_test()
    # et.similarity_test()

