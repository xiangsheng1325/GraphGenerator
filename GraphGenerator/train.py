import scipy.sparse as sp
import GraphGenerator.models.sbm as sbm
import GraphGenerator.models.vgae as vgae
import GraphGenerator.models.graphite as graphite
import networkx as nx
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys, torch, copy, datetime


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


def train_autoencoder(sp_adj, feature, config, model, optimizer):
    norm = sp_adj.shape[0] * sp_adj.shape[0] / float((sp_adj.shape[0] * sp_adj.shape[0] - sp_adj.sum()) * 2)
    pos_weight = torch.tensor(float(sp_adj.shape[0] * sp_adj.shape[0] - sp_adj.sum()) / sp_adj.sum()).to(config.device)
    adj_def = torch.from_numpy(sp_adj.toarray()).to(config.device)
    adj_normalized = sp_normalize(sp_adj, config.device)
    adj_normalized = Variable(adj_normalized).to(config.device)
    training_time = datetime.timedelta()
    for epoch in range(config.train.max_epochs):
        epoch_start = datetime.datetime.now()
        adj_score = model(adj_normalized, feature, device=config.device)
        train_loss = norm * F.binary_cross_entropy_with_logits(adj_score, adj_def,
                                                               pos_weight=pos_weight)
        if config.model.variational:
            kl_div = 0.5/adj_score.size(0)*(1+2*model.logv-model.mean**2-torch.exp(model.logv)**2).sum(1).mean()
            train_loss -= kl_div
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        epoch_time = datetime.datetime.now() - epoch_start
        training_time += epoch_time
        print('[%03d/%d]: loss:%.4f, time per epoch:%.8s'
              % (epoch + 1,
                 config.train.max_epochs,
                 train_loss,
                 str(epoch_time)[-12:]))
    print('### Training Time Consumption:%.8s'
          % str(training_time)[-12:])
    return model


def top_n_indexes(arr, n):
    idx = np.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def topk_adj(adj, k):
    if isinstance(adj, torch.Tensor):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = adj
    assert ((adj_ == adj_.T).all())
    adj_ = (adj_ - np.min(adj_)) / np.ptp(adj_)
    adj_ -= np.diag(np.diag(adj_))
    res = np.zeros(adj.shape)
    tri_adj = np.triu(adj_)
    inds = top_n_indexes(tri_adj, int(k//2))
    for ind in inds:
        i = ind[0]
        j = ind[1]
        res[i, j] = 1.0
        res[j, i] = 1.0
    return res


def infer_autoencoder(sp_adj, feature, config, model, repeat=1):
    generated_graphs = []
    with torch.no_grad():
        adj_normalized = sp_normalize(sp_adj, config.device)
        adj_normalized = Variable(adj_normalized).to(config.device)
        for i in range(repeat):
            adj_score = model(adj_normalized, feature, device=config.device)
            adj = topk_adj(adj_score, k=sp_adj.sum())
            tmp_graph = nx.from_numpy_array(adj)
            generated_graphs.append(tmp_graph)
    return generated_graphs


def train_and_inference(input_data, generator, config=None, repeat=1):
    """
    train model using input graph, and infer new graphs
    :param input_data: graph object, whose type is networkx.Graph
    :param generator: name of graph generator
    :param config: configuration of graph generator
    :param repeat: number of new graphs
    :return: generated graphs
    """
    graphs = []
    if generator in ['sbm', 'dcsbm']:
        graphs = sbm.generate(input_data, generator, repeat)
    elif generator in ['vgae', 'graphite']:
        sp_adj = nx.adjacency_matrix(input_data).astype(np.float32)
        # print("Shape!", sp_adj.shape)
        feature = coo_to_csp(sp.diags(np.array([1. for i in range(sp_adj.shape[0])],
                                                dtype=np.float32)).tocoo()).to(config.device)
        if config.model.variational:
            if generator == 'vgae':
                model = vgae.VGAE(config.model.num_nodes,
                                  config.model.embedding_dim,
                                  config.model.hidden_dim,
                                  act=F.relu,
                                  layers=config.model.num_GNN_layers).to(config.device)
            elif generator == 'graphite':
                model = graphite.GraphiteVAE(config.model.num_nodes,
                                             config.model.hidden_dim,
                                             config.model.embedding_dim,
                                             config.model.decoding_dim,
                                             act=F.relu).to(config.device)
            else:
                model = None
                sys.exit(1)
        else:
            if generator == 'vgae':
                model = vgae.GAE(config.model.num_nodes,
                                 config.model.embedding_dim,
                                 config.model.hidden_dim,
                                 act=F.relu,
                                 layers=config.model.num_GNN_layers).to(config.device)
            elif generator == 'graphite':
                model = graphite.GraphiteAE(config.model.num_nodes,
                                            config.model.hidden_dim,
                                            config.model.embedding_dim,
                                            config.model.decoding_dim,
                                            act=F.relu).to(config.device)
            else:
                model = None
                sys.exit(1)
        optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
        model = train_autoencoder(sp_adj, feature, config, model, optimizer)
        graphs = infer_autoencoder(sp_adj, feature, config, model, repeat=repeat)
    else:
        print("Wrong generator name! Process exit..")
        sys.exit(1)
    return graphs

