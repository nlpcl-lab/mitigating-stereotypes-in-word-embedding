"""
Revised by Huije Lee
Code reference
William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from
Unlabeled Corpora. Proceedings of EMNLP. 2016. (to appear; arXiv:1606.02820).
"""
import random
import gensim
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
from keras import backend as K
from keras.models import Graph
from keras.layers.core import Dense as DenseCore
from keras.layers.core import Lambda as LambdaCore
from keras.optimizers import Optimizer
from keras.constraints import Constraint

from embedding import Embedding
import config


class SimpleSGD(Optimizer):
    def __init__(self, lr=5, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SimpleSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * 0.99
        self.updates = [(self.iterations, self.iterations + 1.)]

        # momentum
        self.weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        for p, g, m in zip(params, grads, self.weights):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SimpleSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Orthogonal(Constraint):
    def __call__(self, p):
        print("here")
        s,u,v = tf.linalg.svd(p)
        return K.dot(u,K.transpose(v))


class DatasetMinibatchIterator:
    def __init__(self, embeddings, positive_seeds, negative_seeds, batch_size=512, **kwargs):
        self.words, embeddings1, embeddings2, labels = [], [], [], []

        def add_examples(word_pairs, label):
            for w1, w2 in word_pairs:
                embeddings1.append(embeddings[w1])
                embeddings2.append(embeddings[w2])
                labels.append(label)
                self.words.append((w1, w2))

        add_examples(combinations(positive_seeds, 2), 1)
        add_examples(combinations(negative_seeds, 2), 1)
        add_examples(product(positive_seeds, negative_seeds), -1)
        self.e1 = np.vstack(embeddings1)
        self.e2 = np.vstack(embeddings2)
        self.y = np.array(labels)

        self.batch_size = batch_size
        self.n_batches = int((self.y.size + self.batch_size - 1) / self.batch_size)

    def shuffle(self):
        perm = np.random.permutation(np.arange(self.y.size))
        self.e1, self.e2, self.y, self.words = \
            self.e1[perm], self.e2[perm], self.y[perm], [self.words[i] for i in perm]

    def __iter__(self):
        for i in range(self.n_batches):
            batch = np.arange(i * self.batch_size, min(self.y.size, (i + 1) * self.batch_size))
            yield {
                'embeddings1': self.e1[batch],
                'embeddings2': self.e2[batch],
                'y': self.y[batch][:, np.newaxis]
            }


def get_model_fi(inputdim, outputdim, regularization_strength=0.01, lr=0.001, cosine=False, **kwargs):
    transformation = DenseCore(inputdim, init='identity',
                           W_constraint=Orthogonal())

    model = Graph()
    model.add_input(name='embeddings1', input_shape=(inputdim,))
    model.add_input(name='embeddings2', input_shape=(inputdim,))
    model.add_shared_node(transformation, name='transformation',
                          inputs=['embeddings1', 'embeddings2'],
                          outputs=['transformed1', 'transformed2'])
    model.add_node(LambdaCore(lambda x: x[:, :outputdim]), input='transformed1', name='projected1')
    model.add_node(LambdaCore(lambda x: -x[:, :outputdim]), input='transformed2', name='negprojected2')

    model.add_node(LambdaCore(lambda x: K.reshape(K.sqrt(K.sum(x * x, axis=1)), (-1, 1))),
                       name='distances', inputs=['projected1', 'negprojected2'], merge_mode='sum')

    model.add_output(name='y', input='distances')
    model.compile(loss={'y': lambda y, d: K.mean(y * d)}, optimizer=SimpleSGD())
    return model



def apply_embedding_transformation(embeddings, positive_seeds, negative_seeds, entity_seeds, notity_seeds,
                                   n_epochs=5, n_dim=10, n2_dim=10, plot=False, plot_points=50, plot_seeds=False,
                                   lexicon2_vocab={}, **kwargs):
    print ("Preparing to learn sense embedding tranformation ... {:.2f}".format(config.whattime()))
    dataset = DatasetMinibatchIterator(embeddings, positive_seeds, negative_seeds, **kwargs)
    model = get_model_fi(embeddings.m.shape[1], n_dim, **kwargs)

    print ("Learning sense embedding transformation")
#    prog = util.Progbar(n_epochs)
    for epoch in range(n_epochs):
        dataset.shuffle()
        loss = 0
        for i, X in enumerate(dataset):
            loss += model.train_on_batch(X)[0] * X['y'].size
            Q, b = model.get_weights()
            model.set_weights([Q, np.zeros_like(b)])
#        prog.update(epoch + 1, exact_values=[('loss', loss / dataset.y.size)])

    Q, b = model.get_weights()
    new_mat = embeddings.m.dot(Q)[:, :n_dim]
    embeddings.m = embeddings.m.dot(Q)[:, n_dim:]
    embeddings.dim = embeddings.m.shape[1]

    print("Preparing to learn entity embedding tranformation ... {:.2f}".format(config.whattime()))
    dataset = DatasetMinibatchIterator(embeddings, entity_seeds, notity_seeds, **kwargs)
    model = get_model_fi(embeddings.m.shape[1], n2_dim, **kwargs)

    print("Learning entity embedding transformation")
    #    prog = util.Progbar(n_epochs)
    for epoch in range(n_epochs):
        dataset.shuffle()
        loss = 0
        for i, X in enumerate(dataset):
            loss += model.train_on_batch(X)[0] * X['y'].size
            Q, b = model.get_weights()
            model.set_weights([Q, np.zeros_like(b)])
    #        prog.update(epoch + 1, exact_values=[('loss', loss / dataset.y.size)])

    print("Complete to entire transformation matrix ... {:.2f}".format(config.whattime()))
    Q, b = model.get_weights()
    my_w2v = gensim.models.KeyedVectors(vector_size=300)
    tmp_vocab = {}
    for i, word in enumerate(embeddings.iw):
        try:
            tmp_vocab[word] = config.Vocab(count=lexicon2_vocab[word], index=embeddings.wi[word])
        except Exception as e:
            tmp_vocab[word] = config.Vocab(count=5, index=embeddings.wi[word])
    my_w2v.vocab = tmp_vocab
    my_w2v.index2word = embeddings.iw
    my_w2v.syn0 = np.column_stack((new_mat, embeddings.m.dot(Q)))
    my_w2v.save_word2vec_format(config.MITIGATED_EMBEDDING_NAME, binary=False)
    # for loading test
    # tmp_w2v = gensim.models.KeyedVectors.load(config.MITIGATED_EMBEDDING_NAME)
    new_mat = np.column_stack((new_mat, embeddings.m.dot(Q)[:, :n2_dim]))

    if plot and n_dim == 2:
        plot_words = positive_seeds + negative_seeds if plot_seeds else \
            [w for w in embeddings if w not in positive_seeds and w not in negative_seeds]
        plot_words = set(random.sample(plot_words, plot_points))
        to_plot = {w: embeddings[w] for w in embeddings if w in plot_words}

        lexicon = config.load_sent_lexicon()
        plt.figure(figsize=(10, 10))
        for w, e in to_plot.items():
            plt.text(e[0], e[1], w, bbox=dict(facecolor='green' if lexicon[w] == 1 else 'red', alpha=0.1))
        xmin, ymin = np.min(np.vstack(to_plot.values()), axis=0)
        xmax, ymax = np.max(np.vstack(to_plot.values()), axis=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show()

    return Embedding(new_mat, embeddings.iw, normalize=n_dim!=1)
