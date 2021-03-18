import os
import sys
import pickle as cp
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import torch, torch.cuda
import torch.optim as optim
from collections import OrderedDict
from GraphGenerator.utils.arg_utils import get_config, set_device
from GraphGenerator.models.bigg_ops.tree_clib.tree_lib import setup_treelib, TreeLib
from GraphGenerator.models.bigg_ops.tree_model import RecurTreeGen


def get_node_dist(graphs):
    num_node_dist = np.bincount([len(gg.nodes) for gg in graphs])
    num_node_dist = num_node_dist / np.sum(num_node_dist)
    return num_node_dist


def sqrtn_forward_backward(model,
                           graph_ids,
                           list_node_starts,
                           num_nodes,
                           blksize,
                           loss_scale,
                           init_states=[None, None],
                           top_grad=None,
                           **kwargs):
    assert len(graph_ids) == 1
    if blksize < 0 or blksize > num_nodes:
        blksize = num_nodes

    prev_states = init_states
    cache_stages = list(range(0, num_nodes, blksize))

    list_caches = []
    for st_delta in cache_stages[:-1]:
        node_st = list_node_starts[0] + st_delta
        with torch.no_grad():
            cur_num = num_nodes - node_st if node_st + blksize > num_nodes else blksize
            _, new_states = model.forward_row_summaries(graph_ids,
                                                        list_node_starts=[node_st],
                                                        num_nodes=cur_num,
                                                        prev_rowsum_states=prev_states,
                                                        **kwargs)
            prev_states = new_states
            list_caches.append(new_states)

    tot_ll = 0.0
    for i in range(len(cache_stages) - 1, -1, -1):
        st_delta = cache_stages[i]
        node_st = list_node_starts[0] + st_delta
        cur_num = num_nodes - node_st if node_st + blksize > num_nodes else blksize
        prev_states = list_caches[i - 1] if i else init_states
        if prev_states[0] is not None:
            for x in prev_states:
                x.requires_grad = True
        ll, cur_states = model.forward_train(graph_ids,
                                             list_node_starts=[node_st],
                                             num_nodes=cur_num,
                                             prev_rowsum_states=prev_states,
                                             **kwargs)
        tot_ll += ll.item()
        loss = -ll * loss_scale
        if top_grad is not None:
            torch.autograd.backward([loss, *cur_states], [None, *top_grad])
        else:
            loss.backward()
        if i:
            top_grad = [x.grad.detach() for x in prev_states]

    return tot_ll, top_grad


def train_bigg(train_graphs, config):
    # print("### Type:", type(train_graphs))
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    set_device(config)
    setup_treelib(config)
    for g in train_graphs:
        TreeLib.InsertGraph(g)
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    config.model.max_num_nodes = max_num_nodes

    model = RecurTreeGen(config).to(config.device)
    if config.train.resume and os.path.isfile(config.train.resume_model_dir):
        print('loading from', config.train.resume_model_dir)
        resume_model_path = os.path.join(config.train.resume_model_dir,
                                         config.train.resume_model_name)
        model.load_state_dict(torch.load(resume_model_path))

    optimizer = optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    indices = list(range(len(train_graphs)))
    if config.train.resume_epoch is None:
        config.train.resume_epoch = 0
    for epoch in range(config.train.resume_epoch, config.train.max_epochs):
        pbar = tqdm(range(config.train.snapshot_epoch))

        optimizer.zero_grad()
        for idx in pbar:
            random.shuffle(indices)
            batch_indices = indices[:config.train.batch_size]

            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            if config.model.blksize < 0 or num_nodes <= config.model.blksize:
                ll, _ = model.forward_train(batch_indices)
                loss = -ll / num_nodes
                loss.backward()
                loss = loss.item()
            else:
                ll = 0.0
                for i in batch_indices:
                    n = len(train_graphs[i])
                    cur_ll, _ = sqrtn_forward_backward(model, graph_ids=[i], list_node_starts=[0],
                                                       num_nodes=n, blksize=config.model.blksize, loss_scale=1.0 / n)
                    ll += cur_ll
                loss = -ll / num_nodes
            if (idx + 1) % config.train.accum_grad == 0:
                if config.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / config.train.snapshot_epoch, loss))

        torch.save(model.state_dict(), os.path.join(config.exp_dir, config.exp_name, 'epoch-%d.ckpt' % (epoch + 1)))
    return model


def infer_bigg(test_graphs, config, model=None):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    set_device(config)
    setup_treelib(config)
    max_num_nodes = max([len(gg.nodes) for gg in test_graphs])
    config.model.max_num_nodes = max_num_nodes
    if model is None:
        model = RecurTreeGen(config).to(config.device)
        for g in test_graphs:
            TreeLib.InsertGraph(g)
    test_model_path = os.path.join(config.test.test_model_dir,
                                   config.test.test_model_name)
    if config.test.load_snapshot and os.path.isfile(config.test.test_model_dir):
        print('loading from', config.test.test_model_dir)
        model.load_state_dict(torch.load(test_model_path))

    # get num nodes dist
    num_node_dist = get_node_dist(test_graphs)
    gen_graphs = []
    with torch.no_grad():
        for _ in tqdm(range(config.test.num_test_gen)):
            num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
            _, pred_edges, _ = model(num_nodes, display=config.test.display)
            for e in pred_edges:
                assert e[0] > e[1]
            pred_g = nx.Graph()
            pred_g.add_edges_from(pred_edges)
            gen_graphs.append(pred_g)
    # print('saving graphs')
    # with open(test_model_path + '.graphs-%s' % str(config.test.greedy_frac), 'wb') as f:
    #     cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
    # print('evaluating')
    return gen_graphs
    # sys.exit(0)
