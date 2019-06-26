# -*- coding: utf-8 -*-'
# Creaetor: Huije Lee (https://github.com/huijelee)
# WikiCorpus (2018-10-24 enwiki): 58,860,232 lxml etree elements, 5,739,304 articles

import glob
import codecs
import time
import re
import json
import numpy as np
import pandas as pd
from sklearn import metrics
from gensim.models import word2vec

# initialization
MODEL_DIR = 'model/'
SOURCE_DIR = 'source/'

VOCAB_LIMIT = 100000
ANNOTATED_VOCAB_LIMIT = 10000
MODEL_NAME = 'wiki'  # 'reddit' 'redditsmall'
BASE_WORD_NUM = '20'
WORD_EMBEDDING_NAME = "D:/PycharmProjects_D/Bias/source/glove.6B.300d.txt"
#WORD_EMBEDDING_NAME = MODEL_DIR + MODEL_NAME + ".w2v.300d"
MITIGATED_EMBEDDING_NAME = MODEL_DIR + "migigated{}_".format(BASE_WORD_NUM) + MODEL_NAME + ".glove.300d"
#WORD_EMBEDDING_NAME = MODEL_DIR + 'my_embedding_{}{}".format(MODEL_NAME, BASE_WORD_NUM)
#MITIGATED_EMBEDDING_NAME = MODEL_DIR + 'my_embedding_{}{}".format(MODEL_NAME, BASE_WORD_NUM)
UNBALANCED_BASE_WORDS = False
RANDOM_BASE_WORDS = False
SAVED_MODEL = False # for polarity_induction_methods (skip learning)

# for base_embedding.py
CONSIDER_GENDER = False
WIKI_DIR = 'D:/dataset/wiki/text_en/'
REDDIT_DIR = 'D:/dataset/reddit/'
MINIMUM_WINDOW_SIZE = 11
start_time = time.time()


def whattime():
    return time.time() - start_time


def load_my_model(fname):
    try:
        print('Loading My Model... in {0:.2f} seconds'.format(whattime()))
        my_model = word2vec.Word2VecKeyedVectors.load_word2vec_format(fname, binary=False)
        #my_model = word2vec.Word2Vec.load(fname + 'w2vf')
        print(my_model)

    except IOError:
        print('No existed model. Training My Model... in {0:.2f} seconds'.format(time.time() - start_time))
        my_model = ''

    print('Success to load My Model... in {0:.2f} seconds'.format(time.time() - start_time))
    return my_model


def change_twitter_tag_simpler(tag):
    if tag == "Noun":
        tag = "N"
    elif tag == "Verb":
        tag = "V"
    elif tag == "Adjective":
        tag = "A"
    elif tag == "Adverb":
        tag = "a"
    elif tag == "Determiner":
        tag = "D"
    elif tag == "Eomi":
        tag = "E"
    elif tag == "Exclamation":
        tag = "e"
    elif tag == "Foreign":
        tag = "F"
    elif tag == "Number":
        tag = "n"
    elif tag == "Hashtag":
        tag = "H"
    elif tag == "Suffix":
        tag = "S"
    elif tag == "PreEomi":
        tag = "E"
    elif tag == "CashTag":
        tag = "c"
    elif tag == "VerbPrefix":
        tag = "a"
    elif tag == "NounPrefix":
        tag = "N"
    elif tag == "Punctuation":
        tag = "P"
    elif tag == "Josa":
        tag = "J"
    elif tag == "Alpha":
        tag = "R"
    elif tag == "ScreenName":
        tag = "s"
    elif tag == "KoreanParticle":
        tag = "NP"
    elif tag == "URL":
        tag = "U"
    elif tag == "Email":
        tag = "M"
    elif tag == "Conjunction":
        tag = "a"
    else:
        print("error: " + tag)
        return tag
    return tag


def twitter_symbol_filter(line):
    """ convert to parenthesis """
    line = re.sub(r'&lt;', '<', line)
    line = re.sub(r'&gt;', '>', line)
    """ remove site link"""
    line = re.sub(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
                  '', line)
    """ remove symbol except @(tag detection in twitter.pos) and _(id symbol such as '@like_bot') """
    line = re.sub('[\t\r\n\f(){}\[\]#$%^&*\-+|`~=<>]+', ' ', line)
    """ remove special symbol, emoticons """
    line = re.sub(r'[^ ㄱ-ㅣ가-힣a-zA-Z0-9.,?!@_]+', '', line)
    return line


# means predicted:X, target:y
def find_optimal_cutoff(predicted, target):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']).pop()


def load_sent_lexicon():
    lexicon_dict = dict()
    with codecs.open(SOURCE_DIR + 'opinion_lexicon.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            tokens = line.split('\t')
            lexicon_dict[tokens[0]] = int(tokens[1])

    return lexicon_dict


def load_entity_lexicon():
    lexicon_dict = dict()
    lexicon_vocab_dict = dict()
    with codecs.open(SOURCE_DIR + 'wiki_vocabs_annotated.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            if i > VOCAB_LIMIT:
                break
            tokens = line.split('\t')
            if i >= ANNOTATED_VOCAB_LIMIT and len(tokens) > 1:
                lexicon_dict[tokens[0]] = 0  # 'evaulate' method doesn't evaluate words with label 0.
                lexicon_vocab_dict[tokens[0]] = int(tokens[1])
            elif len(tokens) == 2:
                lexicon_dict[tokens[0]] = -1
                lexicon_vocab_dict[tokens[0]] = int(tokens[1])
            elif len(tokens) == 3 and tokens[2] == '1':
                lexicon_dict[tokens[0]] = 1
                lexicon_vocab_dict[tokens[0]] = int(tokens[1])
            else:
                continue

    return lexicon_dict, lexicon_vocab_dict


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


class TwitterCorpus(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', 1000
        with codecs.open(self.fname, "r", encoding="utf-8", errors='ignore') as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split())  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                # the last token may have been split in two... keep it for the next iteration
                last_token = text.rfind(' ')
                words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else \
                    ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]


class WikiCorpus(object):
    def __init__(self, only_eng=True):
        self.fnames = glob.glob(WIKI_DIR + '*/wiki_*')
        # self.fnames = glob.glob(WIKI_DIR + 'AA/wiki_0*')
        self.doc_count = 0
        self.line_count = 0
        self.token_count = 0
        self.only_eng = only_eng

    def __iter__(self):
        for fname in self.fnames:
            with codecs.open(fname, "r", encoding="utf-8", errors='ignore') as fin:
                docs = re.split('<.+>', fin.read())
                docs = [doc.strip() for doc in docs if len(doc.strip()) > 1]
                # self.doc_count += len(docs)
                for doc in docs:
                    lines = re.split('[\r\n]+', doc)
                    # self.line_count += len(lines)
                    for line in lines:
                        result = [token for token in re.split('\W', line) if token]
                        if self.only_eng:
                            result = [token for token in result if re.search(r'^[a-zA-Z][a-zA-Z0-9]{0,}$', token)]
                        # self.token_count += len(result)
                        if len(result) >= MINIMUM_WINDOW_SIZE:
                            yield result

    def __str__(self):
        return "WikiCorpus(doc=%d, line=%d, token=%d)" % (self.doc_count, self.line_count, self.token_count)


class RedditCorpus(object):
    def __init__(self, only_eng=True):
        self.fnames = glob.glob(REDDIT_DIR + 'RC_2015-0*')
        #self.fnames = glob.glob(REDDIT_DIR + 'RC_2015-01')
        self.doc_count = 0
        self.line_count = 0
        self.token_count = 0
        self.only_eng = only_eng

    def __iter__(self):
        for fname in self.fnames:
            with codecs.open(fname, "r", encoding="utf-8", errors='ignore') as fin:
                while True:
                    read_line = fin.readline()
                    if read_line:
                        comment_json = json.loads(read_line)
                        self.line_count += 1
                        try:
                            line = comment_json['body']
                        except Exception as e:
                            continue
                        result = [token for token in re.split('\W', line) if token]
                        if self.only_eng:
                            result = [token for token in result if re.search(r'^[a-zA-Z][a-zA-Z0-9]{0,}$', token)]
                        # self.token_count += len(result)
                        if len(result) >= MINIMUM_WINDOW_SIZE:
                            yield result
                    else:
                        break

    def __str__(self):
        return "RedditCorpus(doc=%d, line=%d, token=%d)" % (self.doc_count, self.line_count, self.token_count)


if __name__ == "__main__":
    for datastore in RedditCorpus():
        print(datastore)



