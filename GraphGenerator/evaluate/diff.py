import numpy as np
import powerlaw
import networkx as nx
import igraph,datetime
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


# eval utils
def statistics_degrees(A_in):
    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def squares(g):
    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_square_count(A_in):
    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    degrees = A_in.sum(axis=0).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


def statistics_gini(A_in):
    n = A_in.shape[0]
    degrees = np.array(A_in.sum(axis=0)).flatten()
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    degrees = A_in.sum(axis=0).flatten()
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er


def statistics_cluster_coefficient(A_in):
    G = nx.Graph(A_in)
    return nx.average_clustering(G)


def statistics_compute_cpl(A):
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()
    #return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)]


def compute_graph_statistics(A_in, Z_obs=None):
    A = A_in.copy()

    assert((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)# 0.5s

    # Degree statistics
    statistics['deg_max'] = d_max
    statistics['deg_min'] = d_min
    statistics['deg_mean'] = d_mean

    # node number & edger number
    #statistics['node_num'] = A_graph.number_of_nodes()
    #statistics['edge_num'] = A_graph.number_of_edges()

    # largest connected component
    LCC = statistics_LCC(A)# 33.1s

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)# 0.4s

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)# 0.5s

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)# 4.7s

    # Square count
    statistics['square_count'] = statistics_square_count(A)# 41.5s

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)# 1.1s

    # gini coefficient
    statistics['gini'] = statistics_gini(A)# 0.5s

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)# 3.5s

    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)# unknown

    # Clustering coefficient
    statistics['clustering_coefficient'] = statistics_cluster_coefficient(A)# 8.4s

    # Number of connected components
    #statistics['n_components'] = connected_components(A)[0]

    # if Z_obs is not None:
    #     # inter- and intra-community density
    #     intra, inter = statistics_cluster_props(A, Z_obs)
    #     statistics['intra_community_density'] = intra
    #     statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)# 252.4s

    return statistics


def compute_graph_statistics_short(A_in, Z_obs=None):
    A = A_in.copy()
    assert((A == A.T).all())
    statistics = {}
    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)# 1.1s
    # gini coefficient
    statistics['gini'] = statistics_gini(A)# 0.5s
    statistics['cpl'] = statistics_compute_cpl(A)# 252.4s
    return statistics


def diff_graphs(graphs_ref, graphs_pred):
    diff_d = {}
    for g1 in graphs_ref:
        d1 = compute_graph_statistics(nx.to_numpy_array(g1))
        for g2 in graphs_pred:
            d2 = compute_graph_statistics(nx.to_numpy_array(g2))
            for k in list(d1.keys()):
                tmp = diff_d.get(k, 0.)
                diff_d[k] = tmp + round(abs(d1[k] - d2[k]), 5)
    sample_num = len(graphs_ref)*len(graphs_pred)
    for k in list(d1.keys()):
        tmp = diff_d.get(k, 0.)
        diff_d[k] = tmp/sample_num
    return diff_d


def diff_graphs_short(graphs_ref, graphs_pred):
    diff_d = {}
    for g1 in graphs_ref:
        d1 = compute_graph_statistics_short(nx.to_numpy_array(g1))
        for g2 in graphs_pred:
            d2 = compute_graph_statistics_short(nx.to_numpy_array(g2))
            for k in list(d1.keys()):
                tmp = diff_d.get(k, 0.)
                diff_d[k] = tmp + round(abs(d1[k] - d2[k]), 5)
    sample_num = len(graphs_ref)*len(graphs_pred)
    for k in list(d1.keys()):
        tmp = diff_d.get(k, 0.)
        diff_d[k] = tmp/sample_num
    return diff_d


def preprocess_graph(g):
    g.remove_edges_from(nx.selfloop_edges(g))
    g =g.subgraph(max(nx.connected_components(g), key=len))
    g = nx.convert_node_labels_to_integers(g)
    return g
