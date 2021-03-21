import networkx as nx
import itertools
import math
import random


def empty_graph(num_nodes):
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    return g


def complete_graph(num_nodes):
    g = empty_graph(num_nodes)
    edges = itertools.combinations(range(num_nodes), 2)
    g.add_edges_from(edges)
    return g


def random_graph(num_nodes, p):
    g = empty_graph(num_nodes)
    if p <= 0:
        return g
    if p >= 1:
        return complete_graph(num_nodes)
    n = num_nodes
    w = -1
    lp = math.log(1.0 - p)
    # Nodes in graph are from 0,n-1 (start with v as the second node index).
    v = 1
    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            g.add_edge(v, w)
    return g


def e_r(in_graph, config):
    """
    E-R graph generator
    :param in_graph: referenced graph, type: nx.Graph
    :param config: configure object
    :return: generated graphs, type: list of nx.Graph
    """
    num_edges = in_graph.number_of_edges()
    num_nodes = in_graph.number_of_nodes()
    p = num_edges/(num_nodes*(num_nodes-1)/2)
    out_graphs = []
    for i in range(config.num_gen):
        out_graph = random_graph(num_nodes, p)
        out_graphs.append(out_graph)
    return out_graphs
