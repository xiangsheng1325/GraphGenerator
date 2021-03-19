import pyemd, argparse, pickle
from scipy.linalg import toeplitz
import numpy as np
import networkx as nx
import multiprocessing as mp


def wasserstein_distance(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return emd


def degree_loss(x, n=3, real_g=None, generator='W-S', k=2):
    pred_g = nx.empty_graph()
    if generator == 'W-S':
        pred_g = nx.watts_strogatz_graph(n, k, x)
    real_hist = np.array(nx.degree_histogram(real_g))
    real_hist = real_hist / np.sum(real_hist)
    pred_hist = np.array(nx.degree_histogram(pred_g))
    pred_hist = pred_hist / np.sum(pred_hist)
    loss = wasserstein_distance(real_hist, pred_hist)
    return loss


def grid_search(x_min, x_max, x_step, n, real_g, generator, k=2, repeat=2):
    loss_all = []
    x_list = np.arange(x_min, x_max, x_step)
    for x_test in x_list:
        tmp_loss = 0
        for i in range(repeat):
            tmp_loss += degree_loss(x_test, n=n, real_g=real_g, generator=generator, k=k)
        loss_all.append(tmp_loss)
    x_best = x_list[np.argmin(np.array(loss_all))]
    return x_best, min(loss_all)


def generator_optimization(graph, generator='W-S'):
    graph_node = graph.number_of_nodes()
    graph_edge = graph.number_of_edges()
    k = round(graph_edge/graph_node) + 1
    p_selected = 1.
    print('graph with {} nodes'.format(graph_node))
    n = graph_node
    if generator == 'W-S':
        #loss_all = []
        #parameter_all = []
        p_selected, _ = grid_search(1e-6, 1, 0.01, n, graph, generator, k, 10)
    return n, k, p_selected


def generate_new_graph(parameters, generator, repeat=1):
    graph_list = []
    for i in range(repeat):
        if generator == 'W-S':
            graph_list.append(nx.watts_strogatz_graph(*parameters))
    return graph_list


def w_s(in_graph, config):
    """
    W-S graph generator
    :param in_graph: referenced graph, type: nx.Graph
    :param config: configure object
    :return: generated graphs, type: list of nx.Graph
    """
    parameters = generator_optimization(in_graph, config.model.name)
    return generate_new_graph(parameters, config.model.name, repeat=config.num_gen)
