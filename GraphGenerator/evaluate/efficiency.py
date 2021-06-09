from GraphGenerator.metrics import speed, memory
import networkx as nx
import scipy.sparse as sp
import torch, os, copy
import numpy as np


def coo_to_csp(sp_coo):
    num = sp_coo.shape[0]
    row = sp_coo.row
    col = sp_coo.col
    sp_tensor = torch.sparse.FloatTensor(torch.LongTensor(np.stack([row, col])),
                                         torch.tensor(sp_coo.data),
                                         torch.Size([num, num]))
    return sp_tensor


def sp_normalize(adj_def, device='cpu'):
    """
    :param adj: scipy.sparse.coo_matrix
    :param device: default as cpu
    :return: normalized_adj:
    """
    adj_ = sp.coo_matrix(adj_def)
    adj_ = adj_ + sp.coo_matrix(sp.eye(adj_def.shape[0]), dtype=np.float32)
    rowsum = np.array(adj_.sum(axis=1)).reshape(-1)
    norm_unit = np.float_power(rowsum, -0.5).astype(np.float32)
    degree_mat_inv_sqrt = sp.diags(norm_unit)
    degree_mat_sqrt = copy.copy(degree_mat_inv_sqrt)
    # degree_mat_sqrt = degree_mat_inv_sqrt.to_dense()
    support = adj_.__matmul__(degree_mat_sqrt)
    # support = coo_to_csp(support.tocoo())
    # degree_mat_inv_sqrt = coo_to_csp(degree_mat_inv_sqrt.tocoo())
    adj_normalized = degree_mat_inv_sqrt.__matmul__(support)
    adj_normalized = coo_to_csp(adj_normalized.tocoo())
    return adj_normalized


@speed.time_decorator
def eval_speed(func, args):
    pass


def eval_efficiency(generator, config=None):
    from GraphGenerator.train import train_base as train
    # data_sizes = [100, int(1e+3), int(1e+4), int(1e+5), int(1e+6)]
    data_sizes = [1000]
    # data_sizes = config.eval.num_nodes
    print("The tested graph size is: {}.".format(data_sizes))
    output_data = []
    for size in data_sizes:
        new_g = nx.watts_strogatz_graph(size, 4, 0.)
        new_adj = nx.adjacency_matrix(new_g)
        new_adj = sp.coo_matrix(new_adj)
        # adj_input = coo_to_csp(new_adj)
        print("Start (training and) inferencing graph with {} nodes...".format(size))
        tmp_data = train.train_and_inference(new_g, generator, config=config)
        if isinstance(tmp_data, list):
            output_data.extend(tmp_data)
        else:
            output_data.append(tmp_data)
    return output_data


if __name__ == '__main__':
    conf_name = "config/bigg.yaml"
    from GraphGenerator.utils.arg_utils import get_config, set_device
    config = get_config(conf_name)
    set_device(config)
    out = eval_efficiency("bigg", config)
    #