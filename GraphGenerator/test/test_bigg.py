from tqdm import tqdm
import torch
import torch.optim as optim

import numpy as np
import random
import networkx as nx
from GraphGenerator.utils.arg_utils import get_config, set_device
from GraphGenerator.models.bigg.tree_clib.tree_lib import setup_treelib, TreeLib
from GraphGenerator.models.bigg.tree_model import RecurTreeGen


def bigg_test(args):
    config = get_config(args.config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    set_device(config)
    setup_treelib(config)

    train_graphs = [nx.barabasi_albert_graph(10, 2)]
    TreeLib.InsertGraph(train_graphs[0])
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    config.model.max_num_nodes = max_num_nodes

    model = RecurTreeGen(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    for i in range(2):
        optimizer.zero_grad()
        ll, _ = model.forward_train([0])
        loss = -ll / max_num_nodes
        print('iter', i, 'loss', loss.item())
        loss.backward()
        optimizer.step()
