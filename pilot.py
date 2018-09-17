# -*- coding: utf-8 -*-
# Emotional_Word_Dictionary_RES_v1.2: 정규표현식 ^[a-zA-Z0-9]+\t[a-zA-Z0-9_]+\t[가-힣]+/[a-zA-Z]+\t 를 통해 오류줄 색인 가능.

import json, codecs, time, re
import config
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


class EmbeddingTester(object):
    def __init__(self, is_selected_gender_vocab=False):
        self.w2v_fname = config.MODEL_DIR + 'w2v_{0}_sg_300_hs0_neg10_sampled_it10.model'.format(MODEL_NAME)
        self.fasttext_fname = config.MODEL_DIR + 'fasttext_{0}_sg_300_hs0_neg10_sampled_it10.model'.format(MODEL_NAME)
        self.w2v_model = self.load_w2v_model(self.w2v_fname)
        self.fasttext_model = self.load_fasttext_model(self.fasttext_fname)
        if is_selected_gender_vocab:
            self.gender_vocab = self.get_selected_gender_vocab()
        else:
            self.collect_gender_vocab(self.w2v_model)
        self.sentiment_vocab = self.get_sentiment_vocab(debug_mode=False)

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

    def get_selected_gender_vocab(self):
        """
        Return gender vocab(need the collected and 'selected' gender vocab in the directory)
        :return: gender vocab (dict - 0: woman, 1: man)
        """
        with codecs.open(COLLECTED_DATASET_DIR + 'gender_vocab_manuallyselected.txt'.format(MODEL_NAME), "r", encoding='utf-8',
                         errors='ignore') as read_file:
            gender_vocab = OrderedDict()
            for line in read_file.read().splitlines():
                if len(line) > 1:
                    tokens = line.split('\t')
                    gender_vocab[tokens[2]] = tokens[0]

        return gender_vocab

    def get_sentiment_vocab(self, debug_mode=False):
        """
        :param debug_mode: print log or not
        :return: sentiment_vocab (dict keys - positive, negative)
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
                            sentiment_vocab[tokens[7]] = simpler_token

        return sentiment_vocab

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

    def find_most_dist(self):
        """
        need: gender_vocab, sentiment_vocab
        :return: only print
        """
        self.w2v_model.wv.vocab


    def similarity_test(self):
        # Test - 유사도 분석
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
        word_group1 = ['남자/N', '남/N', '남성/N', '아들/N', '형/N', '오빠/N', '아빠/N', '아버지/N', '소년/N',
                       '남자친구/N', '게이/N', '남자배우/N', '남동생/N', '왕자/N', '남학생/N', '미남/N', '장남/N',
                       '손자/N', '남신/N', '남친/N', '시아버지/N', '여심/N', '남자싱글/N', '꽃미남/N', '남창/N',
                       '친오빠/N', '남중/N', '큰아들/N', '남성혐오/N', '남성호르몬/N', '득남/N', '아비/N',
                       '그남자/N', '맏아들/N', '큰아들/N', '막내아들/N', '친형/N', '친아버지/N', '외동아들/N',
                       '남탕/N', '품절남/N', '선남/N', '미소년/N', '새아버지/N', '예비아빠/N', '대장부/N', '차남/N']
        word_group2 = ['여자/N', '여/N', '여성/N', '딸/N', '언니/N', '누나/N', '엄마/N', '어머니/N', '소녀/N',
                       '여자친구/N', '레즈/N', '여배우/N', '여동생/N', '공주/N', '여학생/N', '미녀/N', '장녀/N',
                       '손녀/N', '여신/N', '여친/N', '시어머니/N', '남심/N', '여자싱글/N', '꽃미녀/N', '여창/N',
                       '친언니/N', '여중/N', '큰딸/N', '여성혐오/N', '여성호르몬/N', '득녀/N', '어미/N',
                       '그녀/N', '맏딸/N', '큰딸/N', '막내딸/N', '친언니/N', '친어머니/N', '외동딸/N',
                       '여탕/N', '품절녀/N', '선녀/N', '미소녀/N', '새엄마/N', '예비신부/N', '여장부/N', '차녀/N']
        gender_diff_vec_list = []
        sentiment_invocab_list = []

        for word1, word2 in zip(word_group1, word_group2):
            print(word1, word2, self.w2v_model.most_similar([word1], negative=[word2]))
            gender_diff_vec_list.append(self.w2v_model[word1] - self.w2v_model[word2])

        for word in self.sentiment_vocab['positive'] + self.sentiment_vocab['negative']:
            if word in self.w2v_model.wv.vocab:
                print(word)
                sentiment_invocab_list.append(word)

        print('sentiment vocab in w2v_model is {0} in total {1}'.format(len(sentiment_invocab_list), len(self.sentiment_vocab['positive'] + self.sentiment_vocab['negative'])))


        gender_diff_vec_list = []


if __name__ == '__main__':
    # First, is_selected_gender_vocab=False
    # collect_gender_vocab(w2v_model)
    # Second,
    # after manually selecting gender_vocab with changing file name 'gender_vocab_manuallyselected.txt'
    # do is_selected_gender_vocab=True
    et = EmbeddingTester(is_selected_gender_vocab=True)
    et.definition_test()
    # et.similarity_test()

