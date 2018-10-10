# -*- coding: utf-8 -*-

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

