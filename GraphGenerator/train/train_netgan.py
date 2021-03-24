from GraphGenerator.models.netgan import *
# import tensorflow as tf
from GraphGenerator.utils.arg_utils import set_device
import tensorflow.compat.v1 as tf
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import time, os, pickle
import networkx as nx
import multiprocessing as mp


def score_to_graph(_rws, _n):
    scores_mat = score_matrix_from_random_walks(_rws, _n).tocsr()
    tmp_graph = graph_from_scores(scores_mat, _A_obs.sum())
    return nx.Graph(tmp_graph)


def train_netgan(input_data, config):
    set_device(config)
    emb_size = config.model.embedding_dim
    l_rate = config.train.lr
    _A_obs = nx.adjacency_matrix(input_data)
    _A_obs = _A_obs - sp.csr_matrix(np.diag(_A_obs.diagonal()))
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]
    val_share = config.train.val_share
    test_share = config.train.test_share
    seed = config.seed
    train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(_A_obs, val_share,
                                                                                            test_share, seed,
                                                                                            undirected=True,
                                                                                            connected=True,
                                                                                            asserts=True)
    train_graph = sp.coo_matrix((np.ones(len(train_ones)), (train_ones[:, 0], train_ones[:, 1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()
    rw_len = config.model.rw_len
    batch_size = config.train.batch_size

    walker = RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)

    walker.walk().__next__()
    netgan = NetGAN(_N, rw_len, walk_generator=walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                    W_down_discriminator_size=emb_size, W_down_generator_size=emb_size,
                    l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, batch_size=batch_size,
                    generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=l_rate)
    stopping_criterion = config.train.stopping_criterion

    assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

    if stopping_criterion == "val":  # use val criterion for early stopping
        stopping = None
    elif stopping_criterion == "eo":  # use eo criterion for early stopping
        stopping = 0.5  # set the target edge overlap here
    else:
        stopping = None
    eval_iter = config.train.eval_iter
    display_iter = config.train.display_iter

    log_dict = netgan.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,
                            eval_every=eval_iter, plot_every=display_iter, max_patience=20, max_iters=200000)

    sample_many = netgan.generate_discrete(10000, reuse=True)

    samples = []

    for _ in range(config.test.sample_num):
        if (_ + 1) % 1000 == 0:
            print(_ + 1)
        samples.append(sample_many.eval({netgan.tau: 0.5}))

    rws = np.array(samples).reshape([-1, rw_len])
    pool = mp.Pool(processes=5)
    args_all = [(rws, _N) for i in range(config.test.num_gen)]
    results = [pool.apply_async(score_to_graph, args=args) for args in args_all]
    graphs = [p.get() for p in results]
    return graphs

