# -*- coding: utf-8 -*-'
# Creaetor: Huije Lee (https://github.com/huijelee)
# WikiCorpus (2018-10-24 enwiki): 58,860,232 lxml etree elements, 5,739,304 articles

import pickle as pkl
import glob
import codecs
import time
import word2vec
import re
from collections import defaultdict, Counter
from copy import deepcopy
from gensim import models
from konlpy.tag import Twitter; twitter = Twitter()

MORPHEME = False

if MORPHEME:
    MODEL_DIR = 'model_gensim\\'
else:
    MODEL_DIR = 'model_gensim_no_morph\\'
if MORPHEME:
    COLLECTED_FNAME_NEWS = 'source\\articles_2018.txt'
    COLLECTED_FNAME_TWITTER = 'source\\twitter_all.txt'
else:
    COLLECTED_FNAME_NEWS = 'source\\articles_2018_no_morph.txt'
    COLLECTED_FNAME_TWITTER = 'source\\twitter_all_no_morph.txt'
    COLLECTED_FNAME_WIKI = 'D:\\dataset\\wiki\\en.txt'

WIKI_DIR = 'D:/dataset/wiki/text_en/'
MINIMUM_WINDOW_SIZE = 11
start_time = time.time()


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


def string_filter(line, morph=True):
    # option: remain Korean, Korean Particle, alpha(R), and Email(M)
    # morph = sentence contain morpheme tags(e.g. ABC/N Act/V ...)
    lines = ""
    line = twitter_symbol_filter(line)
    for (word, tag) in twitter.pos(line):
        tag = change_twitter_tag_simpler(tag)
        if tag == 'P' or tag == 'F' or tag == 'H' or tag == 'c' or tag == 'e' or tag == 's' or tag == 'U' or tag == 'M':
            continue
        if morph:
            lines = lines + word + "/" + tag + " "
        else:
            lines = lines + word + " "
    if len(lines.strip()) > 1:
        return lines.strip() + "\n"
    else:
        return ''


# iter 호출시 1000문장씩 배출하도록 중재
class NewsCorpus(object):
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
    def __init__(self):
        #self.fnames = glob.glob(WIKI_DIR + '*/wiki_*')
        self.fnames = glob.glob(WIKI_DIR + 'AA/wiki_0*')
        self.doc_count = 0
        self.line_count = 0
        self.token_count = 0

    def __iter__(self):
        sentence, rest, max_sentence_length = [], '', 1000
        for fname in self.fnames:
            with codecs.open(fname, "r", encoding="utf-8", errors='ignore') as fin:
                docs = re.split('<.+>', fin.read())
                docs = [doc.strip() for doc in docs if len(doc.strip()) > 1]
                # self.doc_count += len(docs)
                for doc in docs:
                    lines = re.split('[\r\n]+', doc)
                    # self.line_count += len(lines)
                    for line in lines:
                        """
                        # something about handling very long line
                        elif len(line) > 8192:
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
                        """
                        result = [token for token in re.split('\W', line) if token]
                        # self.token_count += len(result)
                        if len(result) > MINIMUM_WINDOW_SIZE:
                            yield [token for token in re.split('\W', line) if token]

    def __str__(self):
        return "WikiCorpus(doc=%d, line=%d, token=%d)" % (self.doc_count, self.line_count, self.token_count)

class CorpusCollector(object):
    def __init__(self, save_name, corpus='twitter', encoding='utf-8', progress=1000, min_count=5, analyze_mode=True, debug_mode=False):
        self.save_name = save_name
        self.encoding = encoding
        self.progress = progress
        self.min_count = min_count          # no use in this class
        self.debug_mode = debug_mode
        if corpus == 'twitter':
            self.collect_twitter_corpus()
        elif corpus == 'news':
            self.collect_news_corpus()
        else:
            print("Wrong corpus type input.")
        self.vocab = {}
        self.ngram = Counter()
        if analyze_mode:
            self.analyze_corpus()

    # (string, encoding) -> (list of line)
    def _load_news_txt(self, fname, encoding='utf-8'):
        with codecs.open(fname, "r", encoding=encoding, errors='ignore') as f:
            docs = [doc.strip() for doc in f]
        return docs

    def check_ngram(self, word, position, topk=5):
        filtered_ngrams = filter(lambda x: x[0][position][:len(word)] == word, self.ngram.items())
        filtered_ngrams = sorted(filtered_ngrams, key=lambda x: -x[1])
        print(filtered_ngrams[:topk])

    def bigram_mikolov(self, delta):
        bigram_mikolov_scores = {}
        for ngram, count in self.ngram.items():
            if not (len(ngram) == 2) or count <= delta:
                continue
            score = (count - delta) / (self.vocab[ngram[0]] * self.vocab[ngram[1]])
            bigram_mikolov_scores[ngram] = score

        for ngram, score in sorted(bigram_mikolov_scores.items(), key=lambda x: -x[1])[:20]:
            print('ngram = {} / score = {:.4} / count = {}'.format(ngram, score, self.ngram[ngram]))

        return bigram_mikolov_scores

    def bigram_mikolov_test(self):
        # require list: unigram and bigram data
        assert len(list(self.vocab)) > 2 and len(list(self.ngram))

        self.bigram_mikolov(delta=100)
        print('--------------------------------------------')
        self.bigram_mikolov(delta=1000)

    def collect_news_corpus(self):
        # Step 1: Collect news corpus
        raw_fnames = glob.glob('../dataset/news dataset/*/articles_*.txt')
        print('Step 1 - Collect {0} news corpus in {1:.4} seconds'.format(len(raw_fnames), time.time() - start_time))

        # Step 2: Save collected corpus
        with codecs.open(self.save_name, "w", encoding=self.encoding, errors='ignore') as f:
            # Lookup fnames
            for raw_fname in raw_fnames:
                fname = raw_fname.split('/')[-1]    # e.g. fname = 'article_0_0.txt'
                try:
                    lines = self._load_news_txt(raw_fname)
                    if MORPHEME:
                        f.write('\n'.join(lines) + '\n')
                    else:
                        f.write('\n'.join([' '.join([word[:-2] for word in line.split(' ')]) for line in lines]) + '\n')
                    if self.debug_mode:
                        print('-----: %s has %d lines' % (fname, len(lines)))
                except Exception as e:
                    print('ERROR: %s: %s' % (fname, str(e)))
                    continue

    def collect_twitter_corpus(self):
        # Step 1: Collect twitter corpus
        raw_fnames = glob.glob('../dataset/twitter dataset/*.txt')
        print('Step 1 - Collect {0} twitter corpus in {1:.4} seconds'.format(len(raw_fnames), time.time() - start_time))

        # Step 2: Save collected corpus
        token_count = 0
        sent_count = 0
        with codecs.open(self.save_name, "w", encoding=self.encoding, errors='ignore') as write_file:
            # Lookup fnames
            for i, raw_fname in enumerate(raw_fnames):
                fname = raw_fname.split('\\')[-1]    # e.g. fname = 'article_0_0.txt'
                if i % 1000 == 1:
                    print("--- progress: {0}nd file in {1:.2f} seconds ---".format(i, time.time() - start_time))
                try:
                    with codecs.open(raw_fname, "r", encoding=self.encoding, errors='ignore') as f:
                        sents = [sent.split('\t')[2].strip() for sent in f]
                        sents = sents[1:]   # remove first line('text')
                        if self.debug_mode:
                            print('-----: %s has %d lines' % (fname, len(sents)))
                        for sent in sents:
                            if sent != "\n":
                                if sent_count % 50000 == 1:
                                    print("progress: {0} sents in {1:.2f} seconds".format(sent_count, time.time() - start_time))
                                sent = string_filter(sent, MORPHEME)
                                write_file.write(sent)
                                token_count += len(sent.split(' '))
                                sent_count += 1
                except Exception as e:
                    print('ERROR: %s: %s' % (fname, str(e)))
                    continue

    def analyze_corpus(self):
        vocab = defaultdict(int)
        # Step 3: Analysis of collected corpus
        # ###sents = newsCorpus(self.save_name)
        with codecs.open(self.save_name, "r", encoding=self.encoding, errors='ignore') as f:
            sents = f.read().splitlines()
            for sent_no, sent in enumerate(sents):
                if not sent_no % self.progress and self.debug_mode:
                    print("PROGRESS: at sentence #%i, processed %i words and %i unique words"
                          % (sent_no, sum(vocab.values()), len(vocab)))

                words = sent.split(' ')
                for word in words:
                    vocab[word] += 1

                self.ngram.update(Counter(zip(*[words[i:] for i in range(2)])))
                # ### self.ngram.update(Counter(zip(*[words[i:] for i in range(3)])))

            self.vocab = vocab
            self.ngram = {ngram: count for ngram, count in self.ngram.items() if count >= self.min_count}

            if MORPHEME:
                self.check_ngram('영화/N', position=0)
            else:
                self.check_ngram('영화', position=0)
            self.bigram_mikolov_test()

            print("Stpe 2 - Analysis of corpus in {0:.4} seconds".format(time.time() - start_time))
            print("--- collected %i unique words from a corpus of %i words and %i sentences ---"
                  % (len(vocab), sum(vocab.values()), sent_no + 1))


class VocabCounter(object):
    def __init__(self, fname, corpus_type='NEWS'):
        self.fname = fname
        self.corpus_type = corpus_type

    def count(self):
        if self.corpus_type == 'NEWS':
            sents = NewsCorpus(self.fname)
        else:
            sents = TwitterCorpus(self.fname)
        count = 0
        for i, sent in enumerate(sents):
            if i % 100000 == 0:
                print("current count until {0} sent: {1}".format(i, count))
            count += len(sent)
        print(count-2)


# not used because of time complexity
class W2vManager(object):
    def __init__(self, mtype='sg', hs=0, neg=10, embed_dim=300, sample=10 ^ -5, alpha=0.025, min_alpha=0.0001, seed=1,
                 is_gensim=False):
        """ training setting """
        self.mtype = mtype
        self.hs = hs
        self.neg = neg
        self.embed_dim = embed_dim
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.seed = seed
        self.is_gensim = is_gensim

    @staticmethod
    def save_model(model, saven):
        # delete the huge stupid table again
        table = deepcopy(model.table)
        model.table = None
        # pickle the entire model to disk, so we can load&resume training later
        pkl.dump(model, open(MODEL_DIR + saven, 'wb'), -1)
        # reinstate the table to continue training
        model.table = table

    def train_word2vec_sg(self, corpus='news', it=3, save_every_it=False):
        # load text
        if corpus == 'news':
            sentences = NewsCorpus(COLLECTED_FNAME_NEWS)

        print("Train w2v model in {0:.4} seconds.".format(time.time() - start_time))
        # train the gemsim w2v model
        if self.is_gensim:
            model = models.word2vec.Word2Vec(sentences=sentences, size=self.embed_dim, workers=4, sg=1,
                                             hs=self.hs, sample=self.sample, negative=10, seed=self.seed, iter=it)
            model.save(MODEL_DIR + "w2v_{0}_{1}_{2}_hs{3}_neg{4}{5}_it{6}.model".format(corpus, self.mtype,
                            self.embed_dim, self.hs, self.neg, '_sampled' if self.sample else '', it))
            print("------------ ITERATION {0} in {1:.4} seconds ------------".format(it, time.time() - start_time))
            return
        # gensim w2v model end
        else:
            model = word2vec.Word2Vec(sentences, mtype=self.mtype, hs=self.hs, neg=self.neg,
                                      embed_dim=self.embed_dim, sample=self.sample, seed=self.seed)

        for i in range(1, it):
            print("------------ ITERATION {0} in {1:.4} seconds ------------".format(i, time.time() - start_time))
            if save_every_it:
                self.save_model(model, "w2v_{0}_{1}_{2}_hs{3}_neg{4}{5}_it{6}.model".format(corpus, self.mtype,
                                self.embed_dim, self.hs, self.neg, '_sampled' if self.sample else '', i))
            model.train(sentences, alpha=self.alpha, min_alpha=self.min_alpha)

        self.save_model(model, "w2v_{0}_{1}_{2}_hs{3}_neg{4}{5}_it{6}.model".format(corpus, self.mtype,
                        self.embed_dim, self.hs, self.neg, '_sampled' if self.sample else '', it))
        print("------------ ITERATION {0} in {1:.4} seconds ------------".format(it, time.time() - start_time))

    def load_w2v_model(self, corpus='news', it=3):
        try:
            with open(MODEL_DIR + "w2v_{0}_{1}_{2}_hs{3}_neg{4}{5}_it{6}.model".format(corpus, self.mtype,
                                self.embed_dim, self.hs, self.neg, '_sampled' if self.sample else '', it), 'rb') as f:
                w2v_model = pkl.load(f)
        except IOError:
            print("There is no w2v save model.")
            self.train_word2vec_sg(corpus=corpus, it=it)
            with open(MODEL_DIR + "w2v_{0}_{1}_{2}_hs{3}_neg{4}{5}_it{6}.model".format(corpus, self.mtype,
                                self.embed_dim, self.hs, self.neg, '_sampled' if self.sample else '', it), 'rb') as f:
                w2v_model = pkl.load(f)

        return w2v_model


if __name__ == "__main__":
    """ If you want to get a collected news text"""
    # cc = CorpusCollector(COLLECTED_FNAME_NEWS, corpus='news', debug_mode=False)
    """ If you want to get a collected twitter text"""
    # cc = CorpusCollector(COLLECTED_FNAME_TWITTER, corpus='twitter', analyze_mode=False, debug_mode=False)
    vc = VocabCounter(COLLECTED_FNAME_NEWS, corpus_type='NEWS')
    vc.count()
    """
    # W2vManager uses local word2vec code.
    # not used because of time complexity
    """
    # w2v_manager = W2vManager(mtype='sg', hs=0, neg=10, embed_dim=300, sample=10 ^ -5, alpha=0.025, min_alpha=0.0001,
    #                         is_gensim=True)
    # w2v_model = w2v_manager.load_w2v_model(corpus='news', it=10)

