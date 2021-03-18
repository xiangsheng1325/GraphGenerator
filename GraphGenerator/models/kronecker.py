import GraphGenerator.utils.data_utils as data_utils
import subprocess
import numpy as np
import networkx as nx


def str_to_float(str):
    a, b = str.split('\n')[1:3]
    a = a.strip().split()
    b = b.strip().split()
    return np.array([[float(a[0]), float(a[-1])], [float(b[0]), float(b[-1])]])


def krongen(init_mat, k):
    """
    Kronecker graph generator.
    :param init_mat: initiator, default as a 2*2 shaped matrix
    :param k: iterations, default as int(log2(nodes))
    :return: generated graph with type of 'nx.classes.graph.Graph'
    """
    tmp = np.sum(init_mat)
    edge_num = int(tmp**k)
    og = nx.Graph()
    choice = ['00', '01', '10', '11']
    prob = init_mat/tmp
    # time complexity is O(k*E) < O(N**2)
    for i in range(edge_num):
        x, y = 0, 0
        tmp_rand = np.random.choice(choice, k, True, prob.flatten())
        for j, m_axis in enumerate(tmp_rand):
            add = 2**j
            x += int(m_axis[0])*add
            y += int(m_axis[1])*add
        og.add_edge(x, y)
    return og


def generate(input_graph, config):
    sparse_adj = nx.adjacency_matrix(input_graph)
    k = int(np.log2(sparse_adj.shape[0])) + 1
    init_mat = np.array([[.5625, .1875], [.1875, .0625]])
    if config.exp_name == 'Kronecker':
        tmp_name = "./data/cit_{}.txt".format(config.dataset.name)
        data_utils.adj_to_edgelist(sparse_adj, tmp_name)
        sp_output = subprocess.check_output(
            args=["./GraphGenerator/models/kronecker_ops/examples/kronfit/kronfit",
                  "-i:{}".format(tmp_name),
                  '-m:"{}"'.format(config.model.init_mat),
                  "-o:./{}/{}/{}_to_kronfit.log".format(config.exp_dir, config.exp_name, config.dataset.name),
                  "-gi:100", "-n0:2"]
        )
        utf_output = sp_output.decode('utf8').strip()
        START_STR = "PARAMS"
        output = utf_output[utf_output.find(START_STR):]
        init_mat = str_to_float(output)
        # dump_graphs(args.dataset, 'kronecker', init_mat, k)
    if config.exp_name == 'RMAT':
        edge_num = sparse_adj.sum()/2.
        tmp = np.float_power(edge_num, 1/k)
        init_mat = init_mat*tmp
        # dump_graphs(args.dataset, 'rmat', init_mat, k)
    return [krongen(init_mat, k) for i in range(config.num_gen)]
