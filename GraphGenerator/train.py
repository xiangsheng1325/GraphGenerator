import GraphGenerator.models.sbm as sbm
import GraphGenerator.models.vgae as vgae
import networkx as nx
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import sys


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
    elif generator in ['vgae']:
        sp_adj = nx.adjacency_matrix(input_data).astype(np.float32)
        feature = vgae.coo_to_csp(sp.diags(np.array([1. for i in range(sp_adj.shape[0])],
                                                    dtype=np.float32)).tocoo()).to(config.device)
        model = vgae.GAE(config.model.num_nodes,
                         config.model.embedding_dim,
                         config.model.hidden_dim,
                         act=F.relu).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
        model = vgae.train(sp_adj, feature, config, model, optimizer)
        graphs = vgae.test(sp_adj, feature, config, model, repeat=repeat)
    else:
        print("Wrong generator name! Process exit..")
        sys.exit(1)
    return graphs

