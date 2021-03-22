from GraphGenerator.evaluate.distance import *
from datetime import datetime
import subprocess as sp
import os, time
from scipy.sparse.linalg import eigsh


def closeness_worker(param):
    G, bins = param
    closeness_centrality_list = list(nx.closeness_centrality(G).values())
    hist, _ = np.histogram(
        closeness_centrality_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def closeness_stats(graph_ref_list,
                    graph_pred_list,
                    bins=1000,
                    is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for closeness_hist in executor.map(
                    closeness_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(closeness_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for closeness_hist in executor.map(
                    closeness_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(closeness_hist)
    else:
        for i in range(len(graph_ref_list)):
            cc_list = list(nx.closeness_centrality(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                cc_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            cc_list = list(
                nx.closeness_centrality(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                cc_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0/10, distance_scaling=bins)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=2.0)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing closeness mmd: ', elapsed)
    return mmd_dist


def betweenness_worker(param):
    G, bins, k = param
    try:
        tmp_dict = nx.betweenness_centrality(G, k=k, seed=123)
    except ValueError:
        tmp_dict = nx.betweenness_centrality(G)
    except:
        tmp_dict = nx.betweenness_centrality(G)
    between_centrality_list = list(tmp_dict.values())
    hist, _ = np.histogram(
        between_centrality_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def between_stats(graph_ref_list,
                  graph_pred_list,
                  bins=1000,
                  is_parallel=True,
                  sample_size=200):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for between_hist in executor.map(
                    betweenness_worker, [(G, bins, sample_size) for G in graph_ref_list]):
                sample_ref.append(between_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for between_hist in executor.map(
                    betweenness_worker, [(G, bins, sample_size) for G in graph_pred_list_remove_empty]):
                sample_pred.append(between_hist)
    else:
        for i in range(len(graph_ref_list)):
            bc_list = list(nx.betweenness_centrality(graph_ref_list[i], k=sample_size).values())
            hist, _ = np.histogram(
                bc_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            bc_list = list(
                nx.betweenness_centrality(graph_pred_list_remove_empty[i], k=sample_size).values())
            hist, _ = np.histogram(
                bc_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0, distance_scaling=bins)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=3.0)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing between mmd: ', elapsed)
    return mmd_dist


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=2.0)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


COUNT_START_STR = 'orbit counts:'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges



def orca(graph, tmp_name):
    #tmp_fname = './tmp.txt'
    tmp_fname = './{}'.format(tmp_name)
    f = open(tmp_fname, 'w')
    f.write(
        str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output(
        ['./orca', 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([
        list(map(int,
                 node_cnts.strip().split(' ')))
        for node_cnts in output.strip('\n').split('\n')
    ])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, name_ref, std=50.0):
    prev = datetime.now()
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    i = 0
    for G in graph_ref_list:
        tmp_name = "{}_{}.txt".format(name_ref, i)
        i += 1
        G.remove_edges_from(nx.selfloop_edges(G))
        try:
            orbit_counts = orca(G, tmp_name)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)
    print("orca begin...")
    for G in graph_pred_list:
        G.remove_edges_from(nx.selfloop_edges(G))
        tmp_name = "{}_{}.txt".format(name_ref, i)
        i += 1
        try:
            orbit_counts = orca(G, tmp_name)
        except:
            print("orca error...")
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    # mmd_dist = compute_mmd(
    #     total_counts_ref,
    #     total_counts_pred,
    #     kernel=gaussian,
    #     is_hist=False,
    #     sigma=30.0)

    mmd_dist = compute_mmd(
        total_counts_ref,
        total_counts_pred,
        kernel=gaussian_tv,
        is_hist=False,
        sigma=std)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing orbit mmd: ', elapsed)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list,
                     graph_pred_list,
                     bins=100,
                     is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for clustering_hist in executor.map(clustering_worker,
    #                                       [(G, bins) for G in graph_ref_list]):
    #     sample_ref.append(clustering_hist)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for clustering_hist in executor.map(
    #       clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
    #     sample_pred.append(clustering_hist)

    # check non-zero elements in hist
    # total = 0
    # for i in range(len(sample_pred)):
    #    nz = np.nonzero(sample_pred[i])[0].shape[0]
    #    total += nz
    # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    # mmd_dist = compute_mmd(
    #     sample_ref,
    #     sample_pred,
    #     kernel=gaussian_emd,
    #     sigma=1.0 / 10,
    #     distance_scaling=bins)

    mmd_dist = compute_mmd(
        sample_ref,
        sample_pred,
        kernel=gaussian_tv,
        sigma=2.0)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


def spectral_worker(G):
    eigs = eigsh(nx.normalized_laplacian_matrix(G).todense(), k=G.number_of_nodes(),return_eigenvectors=False)
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2 + 1e-5), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the spectral distributions of two unordered sets of graphs.
      Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
      '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
        #     sample_ref.append(spectral_density)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        #     sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    # print(len(sample_ref), len(sample_pred))

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing spectral mmd: ', elapsed)
    return mmd_dist


def evaluate_mmd(graph_gt, graph_pred, including_spectral=True, name_ref="cora_to_sbm"):
    mmd_degree = degree_stats(graph_gt, graph_pred)
    mmd_clustering = clustering_stats(graph_gt, graph_pred, bins=200)
    mmd_4orbits = orbit_stats_all(graph_gt, graph_pred, name_ref)
    mmd_between = between_stats(graph_gt, graph_pred)
    mmd_close = closeness_stats(graph_gt, graph_pred)

    if including_spectral:
        mmd_spectral = spectral_stats(graph_gt, graph_pred)
    else:
        mmd_spectral = 0.0

    return mmd_degree, mmd_clustering, mmd_4orbits, mmd_between, mmd_close, mmd_spectral


def evaluate_metric(graph_gt, graph_pred, metric="degree", name_ref="cora_to_sbm", std=50.0):
    if metric == "degree":
        mmd = degree_stats(graph_gt, graph_pred)
    elif metric == "clustering":
        mmd = clustering_stats(graph_gt, graph_pred, bins=200)
    else:
        mmd = orbit_stats_all(graph_gt, graph_pred, name_ref, std)
    return mmd


PRINT_TIME = True
if __name__ == '__main__':
    g1 = [nx.connected_watts_strogatz_graph(2000, 4, 0.8) for i in range(1)]
    # g2 = [nx.grid_2d_graph(40, 50) for i in range(1)]
    g2 = [nx.barabasi_albert_graph(2000, 2) for i in range(1)]
    print(evaluate_mmd(g1, g2, including_spectral=True))
