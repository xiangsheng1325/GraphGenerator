import networkx as nx


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
        out_graph = nx.fast_gnp_random_graph(num_nodes, p)
        out_graphs.append(out_graph)
    return out_graphs
