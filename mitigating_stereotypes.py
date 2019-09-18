"""
Revised by Huije Lee
Code reference
William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from
Unlabeled Corpora. Proceedings of EMNLP. 2016. (to appear; arXiv:1606.02820).
"""

import random
import time
import codecs
import numpy as np
import config
import embedding
import base_words
from transformation_method import densify
from sklearn.metrics import roc_auc_score, average_precision_score

start_time = time.time()

DEFAULT_ARGUMENTS = dict(
    # for iterative graph algorithms
    similarity_power=1,
    arccos=True,
    max_iter=50,
    epsilon=1e-6,
    sym=True,

    # for learning embeddings transformation
    n_epochs=50,
    force_orthogonal=False,
    batch_size=100,
    cosine=False,

    ## bootstrap
    num_boots=1,
    n_procs=1,
)


def generate_random_seeds(lexicon, num=10):
    items = lexicon.items()
    pos_items = [item for item in items if item[1] == 1]
    neg_items = [item for item in items if item[1] == -1]
    pos_seeds, _ = zip(*(random.sample(pos_items, num)))
    neg_seeds, _ = zip(*(random.sample(neg_items, num)))
    return pos_seeds, neg_seeds


def generate_random_seeds_imbalanced(lexicon, num=10, num2=10):
    items = lexicon.items()
    pos_items = [item for item in items if item[1] == 1]
    neg_items = [item for item in items if item[1] == -1]
    pos_seeds, _ = zip(*(random.sample(pos_items, num)))
    neg_seeds, _ = zip(*(random.sample(neg_items, num2)))
    return pos_seeds, neg_seeds


def top_n_words(score_dict, eval_words, scope, n=10):
    sorted_list = sorted(score_dict.items(),
                         key=lambda item: -item[1])  # not use np.linalg.norm(item[1]). polarities are ignored.
    top_n_pos, top_n_neg = [], []
    count = 0
    for i, (word, value) in enumerate(sorted_list):
        if count < n and word in eval_words:
            top_n_pos.append((word, value))
            count += 1
    count = 0
    for i, (word, value) in enumerate(sorted_list[::-1]):
        if count < n and word in eval_words:
            top_n_neg.append((word, value))
            count += 1
    print("top{} {} / {}: {} / {}".format(n, scope[0], scope[1], top_n_pos, top_n_neg))



def mitigate_embedding():
    print("Getting evaluation words and embeddings... in {:.2f} sec".format(config.whattime()))
    print("Input: {} / Output: {}".format(config.WORD_EMBEDDING_NAME, config.MITIGATED_EMBEDDING_NAME))
    lexicon = config.load_sent_lexicon()
    eval_words = set(lexicon.keys())
    lexicon2, lexicon2_vocab = config.load_entity_lexicon()
    eval_words2 = set(lexicon2.keys())

    num = int(config.BASE_WORD_NUM)

    if not config.RANDOM_BASE_WORDS:
        positive_seeds, negative_seeds = base_words.sent_seeds(num)
        entity_seeds, notity_seeds = base_words.entity_seeds(num)
    else:
        positive_seeds, negative_seeds = generate_random_seeds(lexicon, num=num)
        if config.UNBALANCED_BASE_WORDS:
            entity_seeds, notity_seeds = generate_random_seeds_imbalanced(lexicon2, num=num, num2=3 * num)
        else:
            entity_seeds, notity_seeds = generate_random_seeds(lexicon2, num=num)

    print('pos / neg = {} / {}'.format(positive_seeds, negative_seeds))
    print('entity / notity = {} / {}'.format(entity_seeds, notity_seeds))

    common_embed = embedding.WordEmbedding(config.WORD_EMBEDDING_NAME,
                                             eval_words.union(positive_seeds).union(negative_seeds).union(eval_words2))

    print("Complete to load original embedding... in {:.2f} sec".format(config.whattime()))
    common_words = set(common_embed.iw)
    eval_words = eval_words.intersection(common_words)
    eval_words2 = eval_words2.intersection(common_words)

    eval_words = [word for word in eval_words if not word in positive_seeds and not word in negative_seeds]
    eval_words2 = [word for word in eval_words2 if not word in entity_seeds and not word in notity_seeds]

    print("Generate a word embedding... in {:.2f} sec".format(time.time() - start_time))
    polarities, entities = run_method(positive_seeds, negative_seeds, entity_seeds, notity_seeds,
                                              common_embed.get_subembed(
                                                  set(eval_words).union(negative_seeds).union(positive_seeds).union(
                                                      eval_words2).union(entity_seeds).union(notity_seeds)),
                                              method=densify,
                                              lr=0.001, regularization_strength=0.001, lexicon2_vocab=lexicon2_vocab,
                                              **DEFAULT_ARGUMENTS)
    with codecs.open(config.MITIGATED_EMBEDDING_INFO, "w", encoding='utf-8', errors='ignore') as f:
        evaluate(polarities, lexicon, eval_words, f, scope=('pos', 'neg'))
        evaluate(entities, lexicon2, eval_words2, f, scope=('entity', 'notity'))

    print("Program end... in {:.2f} sec".format(config.whattime()))


def run_method(positive_seeds, negative_seeds, entity_seeds, notity_seeds, embeddings,
               method=densify, lexicon2_vocab={}, **kwargs):
    positive_seeds = [s for s in positive_seeds if s in embeddings]
    negative_seeds = [s for s in negative_seeds if s in embeddings]
    entity_seeds = [s for s in entity_seeds if s in embeddings]
    notity_seeds = [s for s in notity_seeds if s in embeddings]
    return method(embeddings, positive_seeds, negative_seeds, entity_seeds, notity_seeds, lexicon2_vocab=lexicon2_vocab,
                  **kwargs)


def evaluate(polarities, lexicon, eval_words, f, scope=('pos', 'neg')):
    acc, auc, avg_prec, cutoff = binary_metrics(polarities, lexicon, eval_words)
    space_order = 1
    if auc < 0.5:
        polarities = {word: -1 * polarities[word] for word in polarities}
        acc, auc, avg_prec, cutoff = binary_metrics(polarities, lexicon, eval_words)
        space_order = -1

    top_n_words(polarities, eval_words, scope)
    f.write('{} / {} cutoff:{} with space_order: {}\n'.format(scope[0], scope[1], cutoff, space_order))
    print("{} / {} cutoff: {} with space_order: {}".format(scope[0], scope[1], cutoff, space_order))
    print("Binary metrics:")
    print("==============")
    print("Accuracy with optimal threshold: {:.4f}".format(acc))
    print("ROC AUC Score: {:.4f}".format(auc))
    print("Average Precision Score: {:.4f}".format(avg_prec))


def binary_metrics(polarities, lexicon, eval_words, top_perc=None):
    eval_words = [word for word in eval_words if lexicon[word] != 0]
    y_prob, y_true = [], []
    if top_perc:
        polarities = {word: polarities[word] for word in
                      sorted(eval_words, key=lambda w: abs(polarities[w] - 0.5), reverse=True)[
                      :int(top_perc * len(polarities))]}
    else:
        polarities = {word: polarities[word] for word in eval_words}
    for w in polarities:
        y_prob.append(polarities[w])
        y_true.append((1 + lexicon[w]) / 2)

    n = len(y_true)
    ordered_labels = [y_true[i] for i in sorted(range(n), key=lambda i: y_prob[i])]
    positive = sum(ordered_labels)
    cumsum = np.cumsum(ordered_labels)
    best_accuracy = max([(1 + i - cumsum[i] + positive - cumsum[i]) / float(n) for i in range(n)])

    return best_accuracy, roc_auc_score(y_true, y_prob), average_precision_score(y_true,
                                                                                 y_prob), config.find_optimal_cutoff(
        y_prob, y_true)


if __name__ == '__main__':
    random.seed(0)
    mitigate_embedding()
