import numpy as np
import networkx as nx
import itertools as it
from scipy.linalg import toeplitz
import pyemd
import concurrent.futures
import multiprocessing as mp


def rtg_graph(num_edges, num_chars, beta, q, num_timestick=1,
        bipartite=False, self_loop=False, parallel=True):
    if num_chars > 26:
        raise ValueError('Number of characters cannot be greater than 26')

    all_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    all_chars2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    chars = all_chars[:num_chars] + ['#']
    if bipartite:
        chars2 = all_chars2[:num_chars] + ['$']
    else:
        chars2 = all_chars[:num_chars] + ['#']

    keyboard = create_2d_keyboard(num_chars, q, beta)
    edges = []
    graph = nx.Graph()
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for edge in executor.map(edge_kernel, [
                (chars, chars2, keyboard,
                 bipartite, self_loop) for _ in range(num_edges)
            ]):
                edges.append(edge)
    else:
        for _ in range(num_edges):
            edges.append(create_edge(chars, chars2, keyboard,
                                     bipartite, self_loop))
    graph.add_edges_from(edges)
    return graph


def create_2d_keyboard(num_chars, q, beta):
    # assign unequal probabilities to the keys
    p = np.zeros(num_chars + 1)
    p_remaining = 1 - q
    for i in range(num_chars - 1):
        p[i] = np.random.rand() * p_remaining
        p_remaining -= p[i]
    p[num_chars - 1] = p_remaining
    # last key is the seperator
    p[num_chars] = q

    # init the keyboard with indipendant cross product probs
    keyboard = np.outer(p, p)
    # multiply the imbalance factor
    keyboard = keyboard * beta
    # set diagonal to 0
    np.fill_diagonal(keyboard, 0)
    # calculate remaining probabilities for the diagonal
    # such that each row and column sums up to the
    # marginal probability
    remaining_diag = p - keyboard.sum(axis=0)
    dia_idx = np.diag_indices_from(keyboard)
    keyboard[dia_idx] = remaining_diag

    return keyboard


def create_edge(chars, chars2, keyboard, bipartite, self_loop):
    src_finished = False
    dst_finished = False
    src = ''
    dst = ''
    char_combi = np.fromiter(it.product(chars, chars2),
                             dtype='1str,1str')

    if not self_loop and not bipartite:
        # for the first try the key that produces a selfloop
        # on the delimeter is permitted (to reduce the number
        # of selfloops)
        first_try_keyboard = np.copy(keyboard)
        first_try_keyboard[-1, -1] = 0
        first_try_keyboard = first_try_keyboard / first_try_keyboard.sum()
        src, dst = np.random.choice(char_combi, p=first_try_keyboard.flatten())
        if src == '#':
            src_finished = True
        if dst == '#' or dst == '$':
            dst_finished = True

    while not (src_finished and dst_finished):
        s, d = np.random.choice(char_combi, p=keyboard.flatten())
        if not src_finished:
            src += s
        if not dst_finished:
            dst += d
        if s == '#':
            src_finished = True
        if d == '#' or d == '$':
            dst_finished = True

    # if we produced a self loop but they are not allowed
    # we generate a new edge by running the whole function
    # again
    if ((not self_loop) and (src == dst)):
        return create_edge(chars, chars2, keyboard, bipartite, self_loop)
    else:
        return (src, dst)


def edge_kernel(t):
    return create_edge(*t)


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


def degree_loss(x, n=3, real_g=None, generator='RTG', k=2):
    pred_g = nx.empty_graph()
    if generator in ['RTG', 'rtg']:
        pred_g = rtg_graph(n, 26, beta=x, q=k)
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


def generator_optimization(graph, generator='RTG'):
    graph_node = graph.number_of_nodes()
    print('graph with {} nodes'.format(graph_node))
    parameter_temp = 1
    if generator == 'RTG':
        pool = mp.Pool(processes=8)
        edge_num = graph.number_of_edges()
        args_all = [(.09, 1, .1, edge_num, graph, generator, q ** 2 / 100) for q in range(1, 10)]
        results = [pool.apply_async(grid_search, args=args) for args in args_all]
        output = [p.get() for p in results]
        parameter_all = [o[0] for o in output]
        loss_all = [o[1] for o in output]
        idx = np.argmin(np.array(loss_all))
        parameter_temp = parameter_all[int(idx)]
        parameter_temp = (edge_num, 26, parameter_temp, (list(range(1, 10))[int(idx)]) ** 2 / 100)
    return parameter_temp


def generate_new_graph(parameters, generator, repeat=1):
    graph_list = []
    for i in range(repeat):
        if generator in ['rtg', 'RTG']:
            graph_list.append(rtg_graph(*parameters))
    return graph_list


def rtg(in_graph, config):
    """
    RTG graph generator
    :param in_graph: referenced graph, type: nx.Graph
    :param config: configure object
    :return: generated graphs, type: list of nx.Graph
    """
    parameters = generator_optimization(in_graph, config.model.name)
    return generate_new_graph(parameters, config.model.name, repeat=config.num_gen)


if __name__ == '__main__':
    tmp = rtg_graph(5429, 20, 0.09, 0.01, 1)
