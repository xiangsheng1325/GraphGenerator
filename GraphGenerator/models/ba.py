import networkx as nx
import numpy as np


def b_a(in_graph, config):
    """
    B-A graph generator
    :param in_graph: referenced graph, type: nx.Graph
    :param config: configure object
    :return: generated graphs, type: list of nx.Graph
    """
    m = in_graph.number_of_edges()
    n = in_graph.number_of_nodes()
    k = int((n-np.sqrt(n**2-4*m))//2)
    out_graphs = []
    for i in range(config.num_gen):
        out_graph = nx.barabasi_albert_graph(n, k)
        out_graphs.append(out_graph)
    return out_graphs
