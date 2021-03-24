"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Classification Models for Graphs'
by Aleksandar Bojchevski, Oleksandr Shchur, Daniel Z¨¹gner, Stephan G¨¹nnemann
Published at ICML 2018 in Stockholm, Sweden.

Copyright (C) 2018
Daniel Z¨¹gner
Technical University of Munich
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf

import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from matplotlib import pyplot as plt

tf.compat.v1.disable_eager_execution()

import networkx as nx
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import warnings
import pickle as pkl
from matplotlib import pyplot as plt
import igraph
import powerlaw
from numba import jit


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# load cora, citeseer and pubmed dataset
def Graph_load(dataset='cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    # adj = nx.adjacency_matrix(G)
    max_subg = G.subgraph(max(nx.connected_components(G), key=len))
    adj = nx.adj_matrix(max_subg)
    return adj, features, max_subg


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)['arr_0'].item()
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False

    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges

    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = np.array(list(nx.maximal_matching(nx.DiGraph(A))))
                not_in_cover = np.array(list(set(range(N)).difference(hold_edges.flatten())))

                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                min_size = hold_edges.shape[0] + len(not_in_cover)
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))

                d_nic = d[not_in_cover]

                hold_edges_d1 = np.column_stack((not_in_cover[d_nic > 0],
                                                 np.row_stack(map(np.random.choice,
                                                                  A[not_in_cover[d_nic > 0]].tolil().rows))))

                if np.any(d_nic == 0):
                    hold_edges_d0 = np.column_stack(
                        (np.row_stack(map(np.random.choice, A[:, not_in_cover[d_nic == 0]].T.tolil().rows)),
                         not_in_cover[d_nic == 0]))
                    hold_edges = np.row_stack((hold_edges, hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = np.row_stack((hold_edges, hold_edges_d1))

            else:
                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        test_zeros = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        test_zeros = np.row_stack(test_zeros)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """

    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat


@jit(nopython=True)
def random_walk(edges, node_ixs, rwlen, p=1, q=1, n_walks=1):
    N = len(node_ixs)

    walk = []
    prev_nbs = None
    for w in range(n_walks):
        source_node = np.random.choice(N)
        walk.append(source_node)
        for it in range(rwlen - 1):

            if walk[-1] == N - 1:
                nbs = edges[node_ixs[walk[-1]]::, 1]
            else:
                nbs = edges[node_ixs[walk[-1]]:node_ixs[walk[-1] + 1], 1]

            if it == 0:
                walk.append(np.random.choice(nbs))
                prev_nbs = set(nbs)
                continue

            is_dist_1 = []
            for n in nbs:
                is_dist_1.append(int(n in set(prev_nbs)))

            is_dist_1_np = np.array(is_dist_1)
            is_dist_0 = nbs == walk[-2]
            is_dist_2 = 1 - is_dist_1_np - is_dist_0

            alpha_pq = is_dist_0 / p + is_dist_1_np + is_dist_2 / q
            alpha_pq_norm = alpha_pq / np.sum(alpha_pq)
            rdm_num = np.random.rand()
            cumsum = np.cumsum(alpha_pq_norm)
            nxt = nbs[np.sum(1 - (cumsum > rdm_num))]
            walk.append(nxt)
            prev_nbs = set(nbs)
    return np.array(walk)


class RandomWalker:
    """
    Helper class to generate random walks on the input adjacency matrix.
    """

    def __init__(self, adj, rw_len, p=1, q=1, batch_size=128):
        self.adj = adj
        # if not "lil" in str(type(adj)):
        #    warnings.warn("Input adjacency matrix not in lil format. Converting it to lil.")
        #    self.adj = self.adj.tolil()

        self.rw_len = rw_len
        self.p = p
        self.q = q
        self.edges = np.array(self.adj.nonzero()).T
        self.node_ixs = np.unique(self.edges[:, 0], return_index=True)[1]
        self.batch_size = batch_size

    def walk(self):
        while True:
            yield random_walk(self.edges, self.node_ixs, self.rw_len, self.p, self.q, self.batch_size).reshape(
                [-1, self.rw_len])


def edge_overlap(A, B):
    """
    Compute edge overlap between input graphs A and B, i.e. how many edges in A are also present in graph B. Assumes
    that both graphs contain the same number of edges.

    Parameters
    ----------
    A: sparse matrix or np.array of shape (N,N).
       First input adjacency matrix.
    B: sparse matrix or np.array of shape (N,N).
       Second input adjacency matrix.

    Returns
    -------
    float, the edge overlap.
    """

    return ((A == B) & (A == 1)).sum()


def graph_from_scores(scores, n_edges):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.

    Parameters
    ----------
    scores: np.array of shape (N,N)
            The input transition scores.
    n_edges: int
             The desired number of edges in the target graph.

    Returns
    -------
    target_g: symmettic binary sparse matrix of shape (N,N)
              The assembled graph.

    """

    if len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0

    target_g = np.zeros(scores.shape)  # initialize target graph
    scores_int = scores.toarray().copy()  # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)  # The row sum over the scores.

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N):  # Iterate the nodes in random order

        row = scores_int[n, :].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1

    diff = np.round((n_edges - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g


def symmetric(directed_adjacency, clip_to_one=True):
    """
    Symmetrize the input adjacency matrix.
    Parameters
    ----------
    directed_adjacency: sparse matrix or np.array of shape (N,N)
                        Input adjacency matrix.
    clip_to_one: bool, default: True
                 Whether the output should be binarized (i.e. clipped to 1)

    Returns
    -------
    A_symmetric: sparse matrix or np.array of the same shape as the input
                 Symmetrized adjacency matrix.

    """

    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees + .0001) / (2 * float(m))))
    return H_er


def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:, None].dot(counts[None, :])
        if normalize:
            blocks_outer = np.multiply(block, 1 / blocks_outer)
        return blocks_outer

    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1 - np.eye(in_blocks.shape[0])).mean()
    return diag_mean, offdiag_mean


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """

    A = A_in.copy()

    assert ((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]

    if Z_obs is not None:
        # inter- and intra-community density
        intra, inter = statistics_cluster_props(A, Z_obs)
        statistics['intra_community_density'] = intra
        statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics


class NetGAN:
    """
    NetGAN class, an implicit generative model for graphs using random walks.
    """

    def __init__(self, N, rw_len, walk_generator, generator_layers=[40], discriminator_layers=[30],
                 W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128, noise_dim=16,
                 noise_type="Gaussian", learning_rate=0.0003, disc_iters=3, wasserstein_penalty=10,
                 l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5,
                 temperature_decay=1 - 5e-5, seed=15, gpu_id=0, use_gumbel=True, legacy_generator=False):
        """
        Initialize NetGAN.

        Parameters
        ----------
        N: int
           Number of nodes in the graph to generate.
        rw_len: int
                Length of random walks to generate.
        walk_generator: function
                        Function that generates a single random walk and takes no arguments.
        generator_layers: list of integers, default: [40], i.e. a single layer with 40 units.
                          The layer sizes of the generator LSTM layers
        discriminator_layers: list of integers, default: [30], i.e. a single layer with 30 units.
                              The sizes of the discriminator LSTM layers
        W_down_generator_size: int, default: 128
                               The size of the weight matrix W_down of the generator. See our paper for details.
        W_down_discriminator_size: int, default: 128
                                   The size of the weight matrix W_down of the discriminator. See our paper for details.
        batch_size: int, default: 128
                    The batch size.
        noise_dim: int, default: 16
                   The dimension of the random noise that is used as input to the generator.
        noise_type: str in ["Gaussian", "Uniform], default: "Gaussian"
                    The noise type to feed into the generator.
        learning_rate: float, default: 0.0003
                       The learning rate.
        disc_iters: int, default: 3
                    The number of discriminator iterations per generator training iteration.
        wasserstein_penalty: float, default: 10
                             The Wasserstein gradient penalty applied to the discriminator. See the Wasserstein GAN
                             paper for details.
        l2_penalty_generator: float, default: 1e-7
                                L2 penalty on the generator weights.
        l2_penalty_discriminator: float, default: 5e-5
                                    L2 penalty on the discriminator weights.
        temp_start: float, default: 5.0
                    The initial temperature for the Gumbel softmax.
        min_temperature: float, default: 0.5
                         The minimal temperature for the Gumbel softmax.
        temperature_decay: float, default: 1-5e-5
                           After each evaluation, the current temperature is updated as
                           current_temp := max(temperature_decay*current_temp, min_temperature)
        seed: int, default: 15
              Random seed.
        gpu_id: int or None, default: 0
                The ID of the GPU to be used for training. If None, CPU only.
        use_gumbel: bool, default: True
                Use the Gumbel softmax trick.

        legacy_generator: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks.
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.

        """

        self.params = {
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            'Generator_Layers': generator_layers,
            'Discriminator_Layers': discriminator_layers,
            'W_Down_Generator_size': W_down_generator_size,
            'W_Down_Discriminator_size': W_down_discriminator_size,
            'l2_penalty_generator': l2_penalty_generator,
            'l2_penalty_discriminator': l2_penalty_discriminator,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'Wasserstein_penalty': wasserstein_penalty,
            'temp_start': temp_start,
            'min_temperature': min_temperature,
            'temperature_decay': temperature_decay,
            'disc_iters': disc_iters,
            'use_gumbel': use_gumbel,
            'legacy_generator': legacy_generator
        }

        assert rw_len > 1, "Random walk length must be > 1."

        tf.set_random_seed(seed)

        self.N = N
        self.rw_len = rw_len

        self.noise_dim = self.params['noise_dim']
        self.G_layers = self.params['Generator_Layers']
        self.D_layers = self.params['Discriminator_Layers']
        self.tau = tf.placeholder(1.0, shape=(), name="temperature")

        # W_down and W_up for generator and discriminator
        self.W_down_generator = tf.get_variable('Generator.W_Down',
                                                shape=[self.N, self.params['W_Down_Generator_size']],
                                                dtype=tf.float32,
                                                #                                                initializer=tf.contrib.layers.xavier_initializer()
                                                initializer=tf.glorot_uniform_initializer()
                                                )

        self.W_down_discriminator = tf.get_variable('Discriminator.W_Down',
                                                    shape=[self.N, self.params['W_Down_Discriminator_size']],
                                                    dtype=tf.float32,
                                                    #                                                    initializer=tf.contrib.layers.xavier_initializer()
                                                    initializer=tf.glorot_uniform_initializer()
                                                    )

        self.W_up = tf.get_variable("Generator.W_up", shape=[self.G_layers[-1], self.N],
                                    dtype=tf.float32,
                                    #                                    initializer=tf.contrib.layers.xavier_initializer()
                                    initializer=tf.glorot_uniform_initializer()
                                    )

        self.b_W_up = tf.get_variable("Generator.W_up_bias", dtype=tf.float32, initializer=tf.zeros_initializer,
                                      shape=self.N)

        self.generator_function = self.generator_recurrent
        self.discriminator_function = self.discriminator_recurrent

        self.fake_inputs = self.generator_function(self.params['batch_size'], reuse=False, gumbel=use_gumbel,
                                                   legacy=legacy_generator)
        self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], reuse=True,
                                                           gumbel=use_gumbel, legacy=legacy_generator)

        # Pre-fetch real random walks
        dataset = tf.data.Dataset.from_generator(walk_generator, tf.int32, [self.params['batch_size'], self.rw_len])
        # dataset_batch = dataset.prefetch(2).batch(self.params['batch_size'])
        dataset_batch = dataset.prefetch(100)
        batch_iterator = dataset_batch.make_one_shot_iterator()
        real_data = batch_iterator.get_next()

        self.real_inputs_discrete = real_data
        self.real_inputs = tf.one_hot(self.real_inputs_discrete, self.N)

        self.disc_real = self.discriminator_function(self.real_inputs)
        self.disc_fake = self.discriminator_function(self.fake_inputs, reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.params['batch_size'], 1, 1],
            minval=0.,
            maxval=1.
        )

        self.differences = self.fake_inputs - self.real_inputs
        self.interpolates = self.real_inputs + (alpha * self.differences)
        self.gradients = tf.gradients(self.discriminator_function(self.interpolates, reuse=True), self.interpolates)[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

        # weight regularization; we omit W_down from regularization
        self.disc_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'Disc' in v.name
                                      and not 'W_down' in v.name]) * self.params['l2_penalty_discriminator']
        self.disc_cost += self.disc_l2_loss

        # weight regularization; we omit  W_down from regularization
        self.gen_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                     if 'Gen' in v.name
                                     and not 'W_down' in v.name]) * self.params['l2_penalty_generator']
        self.gen_cost += self.gen_l2_loss

        self.gen_params = [v for v in tf.trainable_variables() if 'Generator' in v.name]
        self.disc_params = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                   beta2=0.9).minimize(self.gen_cost, var_list=self.gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                    beta2=0.9).minimize(self.disc_cost, var_list=self.disc_params)

        if gpu_id is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.InteractiveSession(config=config)
        self.init_op = tf.global_variables_initializer()

    def generate_discrete(self, n_samples, reuse=True, z=None, gumbel=True, legacy=False):
        """
        Generate a random walk in index representation (instead of one hot). This is faster but prevents the gradients
        from flowing into the generator, so we only use it for evaluation purposes.

        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks.
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.

        Returns
        -------
                The generated random walks, shape [None, rw_len, N]


        """

        return tf.argmax(self.generator_function(n_samples, reuse, z, gumbel=gumbel, legacy=legacy), axis=-1)

    def generator_recurrent(self, n_samples, reuse=None, z=None, gumbel=True, legacy=False):
        """
        Generate random walks using LSTM.
        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks.
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        Returns
        -------
        The generated random walks, shape [None, rw_len, N]

        """

        with tf.variable_scope('Generator') as scope:
            if reuse is True:
                scope.reuse_variables()

            def lstm_cell(lstm_size):
                return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
                # return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size) for size in self.G_layers])
            #            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.G_layers])

            # initial states h and c are randomly sampled for each lstm cell
            if z is None:
                initial_states_noise = make_noise([n_samples, self.noise_dim], self.params['noise_type'])
            else:
                initial_states_noise = z
            initial_states = []

            # Noise preprocessing
            for ix, size in enumerate(self.G_layers):
                if legacy:  # old version to initialize LSTM. new version has less parameters and performs just as good.
                    h_intermediate = tf.layers.dense(initial_states_noise, size,
                                                     name="Generator.h_int_{}".format(ix + 1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(h_intermediate, size, name="Generator.h_{}".format(ix + 1), reuse=reuse,
                                        activation=tf.nn.tanh)

                    c_intermediate = tf.layers.dense(initial_states_noise, size,
                                                     name="Generator.c_int_{}".format(ix + 1),
                                                     reuse=reuse, activation=tf.nn.tanh)
                    c = tf.layers.dense(c_intermediate, size, name="Generator.c_{}".format(ix + 1), reuse=reuse,
                                        activation=tf.nn.tanh)

                else:
                    intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.int_{}".format(ix + 1),
                                                   reuse=reuse, activation=tf.nn.tanh)
                    h = tf.layers.dense(intermediate, size, name="Generator.h_{}".format(ix + 1), reuse=reuse,
                                        activation=tf.nn.tanh)
                    c = tf.layers.dense(intermediate, size, name="Generator.c_{}".format(ix + 1), reuse=reuse,
                                        activation=tf.nn.tanh)
                initial_states.append((c, h))

            state = initial_states
            inputs = tf.zeros([n_samples, self.params['W_Down_Generator_size']])
            outputs = []

            # LSTM tine steps
            for i in range(self.rw_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # Get LSTM output
                output, state = self.stacked_lstm.call(inputs, state)

                # Blow up to dimension N using W_up
                output_bef = tf.matmul(output, self.W_up) + self.b_W_up

                # Perform Gumbel softmax to ensure gradients flow
                if gumbel:
                    output = gumbel_softmax(output_bef, temperature=self.tau, hard=True)
                else:
                    output = tf.nn.softmax(output_bef)

                # Back to dimension d
                inputs = tf.matmul(output, self.W_down_generator)

                outputs.append(output)
            outputs = tf.stack(outputs, axis=1)
        return outputs

    def discriminator_recurrent(self, inputs, reuse=None):
        """
        Discriminate real from fake random walks using LSTM.
        Parameters
        ----------
        inputs: tf.tensor, shape (None, rw_len, N)
                The inputs to process
        reuse: bool, default: None
               If True, discriminator variables will be reused.

        Returns
        -------
        final_score: tf.tensor, shape [None,], i.e. a scalar
                     A score measuring how "real" the input random walks are perceived.

        """

        with tf.variable_scope('Discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            input_reshape = tf.reshape(inputs, [-1, self.N])
            output = tf.matmul(input_reshape, self.W_down_discriminator)
            output = tf.reshape(output, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])])

            def lstm_cell(lstm_size):
                return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            #                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            disc_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size) for size in self.D_layers])
            #            disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])
            output_disc, state_disc = tf.nn.static_rnn(cell=disc_lstm_cell, inputs=tf.unstack(output, axis=1),
                                                       dtype='float32')
            #            output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell, inputs=tf.unstack(output, axis=1),
            #                                                              dtype='float32')

            last_output = output_disc[-1]

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score

    def train(self, A_orig, val_ones, val_zeros, max_iters=50000, stopping=None, eval_transitions=15e6,
              transitions_per_iter=150000, max_patience=5, eval_every=500, plot_every=-1, save_directory="../snapshots",
              model_name=None, continue_training=False, tuning_epoch=False):
        """

        Parameters
        ----------
        A_orig: sparse matrix, shape: (N,N)
                Adjacency matrix of the original graph to be trained on.
        val_ones: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation edges
        val_zeros: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation non-edges
        max_iters: int, default: 50,000
                   The maximum number of training iterations if early stopping does not apply.
        stopping: float in (0,1] or None, default: None
                  The early stopping strategy. None means VAL criterion will be used (i.e. evaluation on the
                  validation set and stopping after there has not been an improvement for *max_patience* steps.
                  Set to a value in the interval (0,1] to stop when the edge overlap exceeds this threshold.
        eval_transitions: int, default: 15e6
                          The number of transitions that will be used for evaluating the validation performance, e.g.
                          if the random walk length is 5, each random walk contains 4 transitions.
        transitions_per_iter: int, default: 150000
                              The number of transitions that will be generated in one batch. Higher means faster
                              generation, but more RAM usage.
        max_patience: int, default: 5
                      Maximum evaluation steps without improvement of the validation accuracy to tolerate. Only
                      applies to the VAL criterion.
        eval_every: int, default: 500
                    Evaluate the model every X iterations.
        plot_every: int, default: -1
                    Plot the generator/discriminator losses every X iterations. Set to None or a negative number
                           to disable plotting.
        save_directory: str, default: "../snapshots"
                        The directory to save model snapshots to.
        model_name: str, default: None
                    Name of the model (will be used for saving the snapshots).
        continue_training: bool, default: False
                           Whether to start training without initializing the weights first. If False, weights will be
                           initialized.

        Returns
        -------
        log_dict: dict
                  A dictionary with the following values observed during training:
                  * The generator and discriminator losses
                  * The validation performances (ROC and AP)
                  * The edge overlap values between the generated and original graph
                  * The sampled graphs for all evaluation steps.

        """

        if stopping == None:  # use VAL criterion
            best_performance = 0.0
            patience = max_patience
            print("**** Using VAL criterion for early stopping ****")

        else:  # use EO criterion
            assert "float" in str(type(stopping)) and stopping > 0 and stopping <= 1
            print("**** Using EO criterion of {} for early stopping".format(stopping))

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if model_name is None:
            # Find the file corresponding to the lowest vacant model number to store the snapshots into.
            model_number = 0
            while os.path.exists("{}/model_best_{}.ckpt".format(save_directory, model_number)):
                model_number += 1
            save_file = "{}/model_best_{}.ckpt".format(save_directory, model_number)
            open(save_file, 'a').close()  # touch file
        else:
            save_file = "{}/{}_best.ckpt".format(save_directory, model_name)
        print("**** Saving snapshots into {} ****".format(save_file))

        if not continue_training:
            print("**** Initializing... ****")
            self.session.run(self.init_op)
            print("**** Done.           ****")
        else:
            print("**** Continuing training without initializing weights. ****")

        # Validation labels
        actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))

        # Some lists to store data into.
        gen_losses = []
        disc_losses = []
        graphs = []
        val_performances = []
        eo = []
        temperature = self.params['temp_start']

        starting_time = time.time()
        saver = tf.train.Saver()

        transitions_per_walk = self.rw_len - 1
        # Sample lots of random walks, used for evaluation of model.
        sample_many_count = int(np.round(transitions_per_iter / transitions_per_walk))
        sample_many = self.generate_discrete(sample_many_count, reuse=True)
        n_eval_walks = eval_transitions / transitions_per_walk
        n_eval_iters = int(np.round(n_eval_walks / sample_many_count))

        print("**** Starting training. ****")
        ref_list = [myint ** 2 for myint in range(1, 500)]
        graph_list = []
        for _it in range(max_iters):

            if _it > 0 and _it % (2500) == 0:
                t = time.time() - starting_time
                print('{:<7}/{:<8} training iterations, took {} seconds so far...'.format(_it, max_iters, int(t)))

            # Generator training iteration
            gen_loss, _ = self.session.run([self.gen_cost, self.gen_train_op],
                                           feed_dict={self.tau: temperature})

            _disc_l = []
            # Multiple discriminator training iterations.
            for _ in range(self.params['disc_iters']):
                disc_loss, _ = self.session.run(
                    [self.disc_cost, self.disc_train_op],
                    feed_dict={self.tau: temperature}
                )
                _disc_l.append(disc_loss)

            gen_losses.append(gen_loss)
            disc_losses.append(np.mean(_disc_l))

            # Evaluate the model's progress.
            if _it > 0 and _it % eval_every == 0:

                # Sample lots of random walks.
                smpls = []
                for _ in range(n_eval_iters):
                    smpls.append(self.session.run(sample_many, {self.tau: 0.5}))

                # Compute score matrix
                gr = score_matrix_from_random_walks(np.array(smpls).reshape([-1, self.rw_len]), self.N)
                gr = gr.tocsr()

                # Assemble a graph from the score matrix
                _graph = graph_from_scores(gr, A_orig.sum())
                # Compute edge overlap
                edge_overlap = edge_overlap(A_orig.toarray(), _graph)
                graphs.append(_graph)
                eo.append(edge_overlap)

                edge_scores = np.append(gr[tuple(val_ones.T)].A1, gr[tuple(val_zeros.T)].A1)

                # Compute Validation ROC-AUC and average precision scores.
                val_performances.append((roc_auc_score(actual_labels_val, edge_scores),
                                         average_precision_score(actual_labels_val, edge_scores)))

                # Update Gumbel temperature
                temperature = np.maximum(
                    self.params['temp_start'] * np.exp(-(1 - self.params['temperature_decay']) * _it),
                    self.params['min_temperature'])

                print("**** Iter {:<6} Val ROC {:.3f}, AP: {:.3f}, EO {:.3f} ****".format(_it,
                                                                                          val_performances[-1][0],
                                                                                          val_performances[-1][1],
                                                                                          edge_overlap / A_orig.sum()))

                if stopping is None:  # Evaluate VAL criterion
                    if np.sum(val_performances[-1]) > best_performance:
                        # New "best" model
                        best_performance = np.sum(val_performances[-1])
                        patience = max_patience
                        _ = saver.save(self.session, save_file)
                    else:
                        patience -= 1

                    if patience == 0:
                        print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                        break
                elif edge_overlap / A_orig.sum() >= stopping:  # Evaluate EO criterion
                    print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
                    break

            if plot_every > 0 and (_it + 1) % plot_every == 0:
                if len(disc_losses) > 10:
                    plt.plot(disc_losses[9::], label="Critic loss")
                    plt.plot(gen_losses[9::], label="Generator loss")
                else:
                    plt.plot(disc_losses, label="Critic loss")
                    plt.plot(gen_losses, label="Generator loss")
                plt.legend()
                plt.show()

        print("**** Training completed after {} iterations. ****".format(_it))
        plt.plot(disc_losses[9::], label="Critic loss")
        plt.plot(gen_losses[9::], label="Generator loss")
        plt.legend()
        plt.show()
        if stopping is None:
            saver.restore(self.session, save_file)
        #### Training completed.
        log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses, 'val_performances': val_performances,
                    'edge_overlaps': eo, 'generated_graphs': graphs}
        return log_dict


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.

    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".

    Returns
    -------
    noise tensor

    """

    if type == "Gaussian":
        noise = tf.random_normal(shape)
    elif type == 'Uniform':
        noise = tf.random_uniform(shape, minval=-1, maxval=1)
    else:
        print("ERROR: Noise type {} not supported".format(type))
    return noise


def sample_gumbel(shape, eps=1e-20):
    """
    Sample from a uniform Gumbel distribution. Code by Eric Jang available at
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    Parameters
    ----------
    shape: Shape of the Gumbel noise
    eps: Epsilon for numerical stability.

    Returns
    -------
    Noise drawn from a uniform Gumbel distribution.

    """
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

