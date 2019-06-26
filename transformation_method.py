"""
Revised by Huije Lee
Code reference
William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from
Unlabeled Corpora. Proceedings of EMNLP. 2016. (to appear; arXiv:1606.02820).
"""
import config, densifier


def densify(embeddings, positive_seeds, negative_seeds, entity_seeds, notity_seeds, lexicon2_vocab={}, **kwargs):
    """
    Adapted from: http://arxiv.org/pdf/1602.07572.pdf
    """
    p_seeds = {word:1.0 for word in positive_seeds}
    n_seeds = {word:1.0 for word in negative_seeds}
    e_seeds = {word: 1.0 for word in entity_seeds}
    ne_seeds = {word: 1.0 for word in notity_seeds}
    n_dim = 1
    n2_dim = 1
    new_embeddings = embeddings
    if config.SAVED_MODEL:
        # on the construction. need to synchronize (save) base words.
        embeddings = config.load_my_model(config.WORD_EMBEDDING_NAME + config.BASE_WORD_NUM)
        polarities = {w:new_embeddings[w][0] for w in embeddings.index2word}
        entities = {w:new_embeddings[w][n_dim:n_dim+n2_dim] for w in embeddings.index2word}
    else:
        new_embeddings = densifier.apply_embedding_transformation(
            embeddings, p_seeds, n_seeds, e_seeds, ne_seeds, n_dim=n_dim, n2_dim=n2_dim, plot=False, lexicon2_vocab=lexicon2_vocab, **kwargs)
        polarities = {w:new_embeddings[w][0] for w in embeddings.iw}
        entities = {w:new_embeddings[w][n_dim:n_dim+n2_dim] for w in embeddings.iw}
    # human entity dim is 0, sentiment dim is 1  (densifier.py:222)
    #polarities = {w: new_embeddings[w][n_dim:n_dim+n2_dim] for w in embeddings.iw}
    #entities = {w:new_embeddings[w][0] for w in embeddings.iw}
    return polarities, entities

