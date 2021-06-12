import scipy.sparse as sp
from GraphGenerator.utils.arg_utils import set_device
import networkx as nx
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from GraphGenerator.metrics.memory import get_peak_gpu_memory, flush_cached_gpu_memory
import numpy as np
import sys, torch, copy, datetime
from GraphGenerator.evaluate.efficiency import coo_to_csp, sp_normalize


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
    :param input_data: input graph(s), whose type is networkx.Graph or list of nx.Graph
    :param generator: name of graph generator
    :param config: configuration of graph generator
    :param repeat: number of new graphs
    :return: generated graphs
    """
    # graphs = []
    if generator in ['e-r', 'w-s', 'b-a', 'E-R', 'W-S', 'B-A']:
        import GraphGenerator.models.er as er
        import GraphGenerator.models.ws as ws
        import GraphGenerator.models.ba as ba
        tmp_name = generator.lower()
        model_name = "{}.{}".format(tmp_name.replace('-', ''), tmp_name.replace('-', '_'))
        graphs = eval(model_name)(input_data, config)
    elif generator in ['rtg', 'RTG', 'bter', 'BTER']:
        import GraphGenerator.models.rtg as rtg
        import GraphGenerator.models.bter as bter
        model_name = "{}.{}".format(generator, generator)
        graphs = eval(model_name)(input_data, config)
    elif generator in ['sbm', 'dcsbm']:
        import GraphGenerator.models.sbm as sbm
        graphs = sbm.generate(input_data, generator, repeat)
    elif generator in ['rmat', 'kronecker']:
        import GraphGenerator.models.kronecker as kronecker
        import GraphGenerator.models.rmat as rmat
        graphs = eval(generator).generate(input_data, config)
    elif generator in ['vgae', 'graphite']:
        set_device(config)
        sp_adj = nx.adjacency_matrix(input_data).astype(np.float32)
        # print("Shape!", sp_adj.shape)
        feature = coo_to_csp(sp.diags(np.array([1. for i in range(sp_adj.shape[0])],
                                                dtype=np.float32)).tocoo()).to(config.device)
        if generator == 'vgae':
            import GraphGenerator.models.vgae as vgae
            if config.model.variational:
                model_name = "{}.{}".format(generator, "VGAE")
            else:
                model_name = "{}.{}".format(generator, "GAE")
            model = eval(model_name)(config.model.num_nodes,
                                     config.model.embedding_dim,
                                     config.model.hidden_dim,
                                     act=F.relu,
                                     layers=config.model.num_GNN_layers).to(config.device)
        elif generator == 'graphite':
            import GraphGenerator.models.graphite as graphite
            if config.model.variational:
                model_name = "{}.{}".format(generator, "GraphiteVAE")
            else:
                model_name = "{}.{}".format(generator, "GraphiteAE")
            model = eval(model_name)(config.model.num_nodes,
                                     config.model.hidden_dim,
                                     config.model.embedding_dim,
                                     config.model.decoding_dim,
                                     act=F.relu).to(config.device)
        elif generator == 'sbmgnn':
            import GraphGenerator.models.sbmgnn as sbmgnn
        else:
            # model = None
            sys.exit(1)
        optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
        model = train_autoencoder(sp_adj, feature, config, model, optimizer)
        tmp_memory = get_peak_gpu_memory(device=config.device)
        print("Peak GPU memory reserved in training process: {} MiB".format(tmp_memory//1024//1024))
        flush_cached_gpu_memory()
        graphs = infer_autoencoder(sp_adj, feature, config, model, repeat=repeat)
    elif generator in ['graphrnn', 'gran', 'bigg']:
        import GraphGenerator.train.train_graphrnn as graphrnn
        import GraphGenerator.models.bigg as bigg
        import GraphGenerator.models.gran as gran
        if isinstance(input_data, nx.Graph):
            input_data = [input_data]
        trained_model = eval("{}.train_{}".format(generator, generator))(input_data, config)
        tmp_memory = get_peak_gpu_memory(device=config.device)
        print("Peak GPU memory reserved in training process: {} MiB".format(tmp_memory//1024//1024))
        flush_cached_gpu_memory()
        graphs = eval("{}.infer_{}".format(generator, generator))(input_data, config, trained_model)
    else:
        print("Wrong generator name! Process exit..")
        sys.exit(1)
    return graphs

