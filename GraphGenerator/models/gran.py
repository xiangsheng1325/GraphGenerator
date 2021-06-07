from __future__ import (division, print_function)
import os
import networkx as nx
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import os
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F


EPS = np.finfo(np.float32).eps


class GNN(nn.Module):

    def __init__(self,
                 msg_dim,
                 node_state_dim,
                 edge_feat_dim,
                 num_prop=1,
                 num_layer=1,
                 has_attention=True,
                 att_hidden_dim=128,
                 has_residual=False,
                 has_graph_output=False,
                 output_hidden_dim=128,
                 graph_output_dim=None):
        super(GNN, self).__init__()
        self.msg_dim = msg_dim
        self.node_state_dim = node_state_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_prop = num_prop
        self.num_layer = num_layer
        self.has_attention = has_attention
        self.has_residual = has_residual
        self.att_hidden_dim = att_hidden_dim
        self.has_graph_output = has_graph_output
        self.output_hidden_dim = output_hidden_dim
        self.graph_output_dim = graph_output_dim

        self.update_func = nn.ModuleList([
            nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
            for _ in range(self.num_layer)
        ])

        self.msg_func = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Linear(self.node_state_dim + self.edge_feat_dim,
                              self.msg_dim),
                    nn.ReLU(),
                    nn.Linear(self.msg_dim, self.msg_dim)
                ]) for _ in range(self.num_layer)
        ])

        if self.has_attention:
            self.att_head = nn.ModuleList([
                nn.Sequential(
                    *[
                        nn.Linear(self.node_state_dim + self.edge_feat_dim,
                                  self.att_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.att_hidden_dim, self.msg_dim),
                        nn.Sigmoid()
                    ]) for _ in range(self.num_layer)
            ])

        if self.has_graph_output:
            self.graph_output_head_att = nn.Sequential(*[
                nn.Linear(self.node_state_dim, self.output_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.output_hidden_dim, 1),
                nn.Sigmoid()
            ])

            self.graph_output_head = nn.Sequential(
                *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

    def _prop(self, state, edge, edge_feat, layer_idx=0):
        ### compute message
        state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
        if self.edge_feat_dim > 0:
            edge_input = torch.cat([state_diff, edge_feat], dim=1)
        else:
            edge_input = state_diff

        msg = self.msg_func[layer_idx](edge_input)

        ### attention on messages
        if self.has_attention:
            att_weight = self.att_head[layer_idx](edge_input)
            msg = msg * att_weight

        ### aggregate message by sum
        state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
        scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
        state_msg = state_msg.scatter_add(0, scatter_idx, msg)

        ### state update
        state = self.update_func[layer_idx](state_msg, state)
        return state

    def forward(self, node_feat, edge, edge_feat, graph_idx=None):
        """
          N.B.: merge a batch of graphs as a single graph

          node_feat: N X D, node feature
          edge: M X 2, edge indices
          edge_feat: M X D', edge feature
          graph_idx: N X 1, graph indices
        """

        state = node_feat
        prev_state = state
        for ii in range(self.num_layer):
            if ii > 0:
                state = F.relu(state)

            for jj in range(self.num_prop):
                state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

        if self.has_residual:
            state = state + prev_state

        if self.has_graph_output:
            num_graph = graph_idx.max() + 1
            node_att_weight = self.graph_output_head_att(state)
            node_output = self.graph_output_head(state)

            # weighted average
            reduce_output = torch.zeros(num_graph,
                                        node_output.shape[1]).to(node_feat.device)
            reduce_output = reduce_output.scatter_add(0,
                                                      graph_idx.unsqueeze(1).expand(
                                                          -1, node_output.shape[1]),
                                                      node_output * node_att_weight)

            const = torch.zeros(num_graph).to(node_feat.device)
            const = const.scatter_add(
                0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

            reduce_output = reduce_output / const.view(-1, 1)

            return reduce_output
        else:
            return state


class GRANMixtureBernoulli(nn.Module):
    """ Graph Recurrent Attention Networks """

    def __init__(self, config):
        super(GRANMixtureBernoulli, self).__init__()
        self.config = config
        self.device = config.device
        self.max_num_nodes = config.model.max_num_nodes
        self.hidden_dim = config.model.hidden_dim
        self.is_sym = config.model.is_sym
        self.block_size = config.model.block_size
        self.sample_stride = config.model.sample_stride
        self.num_GNN_prop = config.model.num_GNN_prop
        self.num_GNN_layers = config.model.num_GNN_layers
        self.edge_weight = config.model.edge_weight if hasattr(
            config.model, 'edge_weight') else 1.0
        self.dimension_reduce = config.model.dimension_reduce
        self.has_attention = config.model.has_attention
        self.num_canonical_order = config.model.num_canonical_order
        self.output_dim = 1
        self.num_mix_component = config.model.num_mix_component
        self.has_rand_feat = False  # use random feature instead of 1-of-K encoding
        self.att_edge_dim = 64

        self.output_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component))

        self.output_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component))

        if self.dimension_reduce:
            self.embedding_dim = config.model.embedding_dim
            self.decoder_input = nn.Sequential(
                nn.Linear(self.max_num_nodes, self.embedding_dim))
        else:
            self.embedding_dim = self.max_num_nodes

        self.decoder = GNN(
            msg_dim=self.hidden_dim,
            node_state_dim=self.hidden_dim,
            edge_feat_dim=2 * self.att_edge_dim,
            num_prop=self.num_GNN_prop,
            num_layer=self.num_GNN_layers,
            has_attention=self.has_attention)

        ### Loss functions
        pos_weight = torch.ones([1]) * self.edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction='none')

    def _inference(self,
                   A_pad=None,
                   edges=None,
                   node_idx_gnn=None,
                   node_idx_feat=None,
                   att_idx=None):
        """ generate adj in row-wise auto-regressive fashion """

        B, C, N_max, _ = A_pad.shape
        H = self.hidden_dim
        K = self.block_size
        A_pad = A_pad.view(B * C * N_max, -1)

        if self.dimension_reduce:
            node_feat = self.decoder_input(A_pad)  # BCN_max X H
        else:
            node_feat = A_pad  # BCN_max X N_max

        ### GNN inference
        # pad zero as node feature for newly generated nodes (1st row)
        node_feat = F.pad(
            node_feat, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max

        # create symmetry-breaking edge feature for the newly generated nodes
        att_idx = att_idx.view(-1, 1)

        if self.has_rand_feat:
            # create random feature
            att_edge_feat = torch.zeros(edges.shape[0],
                                        2 * self.att_edge_dim).to(node_feat.device)
            idx_new_node = (att_idx[[edges[:, 0]]] >
                            0).long() + (att_idx[[edges[:, 1]]] > 0).long()
            idx_new_node = idx_new_node.byte().squeeze()
            att_edge_feat[idx_new_node, :] = torch.randn(
                idx_new_node.long().sum(),
                att_edge_feat.shape[1]).to(node_feat.device)
        else:
            # create one-hot feature
            att_edge_feat = torch.zeros(edges.shape[0],
                                        2 * self.att_edge_dim).to(node_feat.device)
            # scatter with empty index seems to cause problem on CPU but not on GPU
            att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
            att_edge_feat = att_edge_feat.scatter(
                1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

        # GNN inference
        # N.B.: node_feat is shared by multiple subgraphs within the same batch
        node_state = self.decoder(
            node_feat[node_idx_feat], edges, edge_feat=att_edge_feat)

        ### Pairwise predict edges
        diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]

        log_theta = self.output_theta(diff)  # B X (tt+K)K
        log_alpha = self.output_alpha(diff)  # B X (tt+K)K
        log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
        log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K

        return log_theta, log_alpha

    def _sampling(self, B):
        """ generate adj in row-wise auto-regressive fashion """
        with torch.no_grad():

            K = self.block_size
            S = self.sample_stride
            H = self.hidden_dim
            N = self.max_num_nodes
            mod_val = (N - K) % S
            if mod_val > 0:
                N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
            else:
                N_pad = N

            A = torch.zeros(B, N_pad, N_pad).to(self.device)
            dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes

            ### cache node state for speed up
            node_state = torch.zeros(B, N_pad, dim_input).to(self.device)

            for ii in range(0, N_pad, S):
                # for ii in range(0, 3530, S):
                jj = ii + K
                if jj > N_pad:
                    break

                # reset to discard overlap generation
                A[:, ii:, :] = .0
                A = torch.tril(A, diagonal=-1)

                if ii >= K:
                    if self.dimension_reduce:
                        node_state[:, ii - K:ii, :] = self.decoder_input(A[:, ii - K:ii, :N])
                    else:
                        node_state[:, ii - K:ii, :] = A[:, ii - S:ii, :N]
                else:
                    if self.dimension_reduce:
                        node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
                    else:
                        node_state[:, :ii, :] = A[:, ii - S:ii, :N]

                node_state_in = F.pad(
                    node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)

                ### GNN propagation
                adj = F.pad(
                    A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)  # B X jj X jj
                adj = torch.tril(adj, diagonal=-1)
                adj = adj + adj.transpose(1, 2)
                edges = [
                    adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
                    for bb in range(B)
                ]
                edges = torch.cat(edges, dim=1).t()

                att_idx = torch.cat([torch.zeros(ii).long(),
                                     torch.arange(1, K + 1)]).to(self.device)
                att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

                if self.has_rand_feat:
                    # create random feature
                    att_edge_feat = torch.zeros(edges.shape[0],
                                                2 * self.att_edge_dim).to(self.device)
                    idx_new_node = (att_idx[[edges[:, 0]]] >
                                    0).long() + (att_idx[[edges[:, 1]]] > 0).long()
                    idx_new_node = idx_new_node.byte().squeeze()
                    att_edge_feat[idx_new_node, :] = torch.randn(
                        idx_new_node.long().sum(), att_edge_feat.shape[1]).to(self.device)
                else:
                    # create one-hot feature
                    att_edge_feat = torch.zeros(edges.shape[0],
                                                2 * self.att_edge_dim).to(self.device)
                    att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
                    att_edge_feat = att_edge_feat.scatter(
                        1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

                node_state_out = self.decoder(
                    node_state_in.view(-1, H), edges, edge_feat=att_edge_feat)
                node_state_out = node_state_out.view(B, jj, -1)

                idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
                idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

                diff = node_state_out[:, idx_row, :] - node_state_out[:, idx_col, :]  # B X (ii+K)K X H
                diff = diff.view(-1, node_state.shape[2])
                log_theta = self.output_theta(diff)
                log_alpha = self.output_alpha(diff)

                log_theta = log_theta.view(B, -1, K, self.num_mix_component)  # B X K X (ii+K) X L
                log_theta = log_theta.transpose(1, 2)  # B X (ii+K) X K X L

                log_alpha = log_alpha.view(B, -1, self.num_mix_component)  # B X K X (ii+K)
                prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
                alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

                prob = []
                for bb in range(B):
                    prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]

                prob = torch.stack(prob, dim=0)
                A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])

            ### make it symmetric
            if self.is_sym:
                A = torch.tril(A, diagonal=-1)
                A = A + A.transpose(1, 2)

            return A

    def forward(self, input_dict):
        """
          B: batch size
          N: number of rows/columns in mini-batch
          N_max: number of max number of rows/columns
          M: number of augmented edges in mini-batch
          H: input dimension of GNN
          K: block size
          E: number of edges in mini-batch
          S: stride
          C: number of canonical orderings
          D: number of mixture Bernoulli

          Args:
            A_pad: B X C X N_max X N_max, padded adjacency matrix
            node_idx_gnn: M X 2, node indices of augmented edges
            node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                          (0 indicates indexing from 0-th row of feature which is
                            always zero and corresponds to newly generated nodes)
            att_idx: N X 1, one-hot encoding of newly generated nodes
                          (0 indicates existing nodes, 1-D indicates new nodes in
                            the to-be-generated block)
            subgraph_idx: E X 1, indices corresponding to augmented edges
                          (representing which subgraph in mini-batch the augmented
                          edge belongs to)
            edges: E X 2, edge as [incoming node index, outgoing node index]
            label: E X 1, binary label of augmented edges
            num_nodes_pmf: N_max, empirical probability mass function of number of nodes

          Returns:
            loss                        if training
            list of adjacency matrices  else
        """
        is_sampling = input_dict[
            'is_sampling'] if 'is_sampling' in input_dict else False
        batch_size = input_dict[
            'batch_size'] if 'batch_size' in input_dict else None
        A_pad = input_dict['adj'] if 'adj' in input_dict else None
        node_idx_gnn = input_dict[
            'node_idx_gnn'] if 'node_idx_gnn' in input_dict else None
        node_idx_feat = input_dict[
            'node_idx_feat'] if 'node_idx_feat' in input_dict else None
        att_idx = input_dict['att_idx'] if 'att_idx' in input_dict else None
        subgraph_idx = input_dict[
            'subgraph_idx'] if 'subgraph_idx' in input_dict else None
        edges = input_dict['edges'] if 'edges' in input_dict else None
        label = input_dict['label'] if 'label' in input_dict else None
        num_nodes_pmf = input_dict[
            'num_nodes_pmf'] if 'num_nodes_pmf' in input_dict else None
        subgraph_idx_base = input_dict[
            "subgraph_idx_base"] if "subgraph_idx_base" in input_dict else None

        N_max = self.max_num_nodes

        if not is_sampling:
            B, _, N, _ = A_pad.shape

            ### compute adj loss
            log_theta, log_alpha = self._inference(
                A_pad=A_pad,
                edges=edges,
                node_idx_gnn=node_idx_gnn,
                node_idx_feat=node_idx_feat,
                att_idx=att_idx)

            num_edges = log_theta.shape[0]

            adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha,
                                              self.adj_loss_func, subgraph_idx, subgraph_idx_base,
                                              self.num_canonical_order)

            return adj_loss
        else:
            A = self._sampling(batch_size)

            ### sample number of nodes
            num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)
            num_nodes = torch.multinomial(
                num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1

            A_list = [
                A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
            ]
            return A_list


def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx, subgraph_idx_base, num_canonical_order,
                           sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean"):
    """
      Compute likelihood for mixture of Bernoulli model

      Args:
        label: E X 1, see comments above
        log_theta: E X D, see comments above
        log_alpha: E X D, see comments above
        adj_loss_func: BCE loss
        subgraph_idx: E X 1, see comments above
        subgraph_idx_base: B+1, cumulative # of edges in the subgraphs associated with each batch
        num_canonical_order: int, number of node orderings considered
        sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp
          i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
          This is equivalent to the original GRAN loss.
        return_neg_log_prob: boolean, if True also return neg log prob
        reduction: string, type of reduction on batch dimension ("mean", "sum", "none")

      Returns:
        loss (and potentially neg log prob)
    """

    num_subgraph = subgraph_idx_base[-1]  # == subgraph_idx.max() + 1
    B = subgraph_idx_base.shape[0] - 1
    C = num_canonical_order
    E = log_theta.shape[0]
    K = log_theta.shape[1]
    assert E % C == 0
    adj_loss = torch.stack(
        [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

    const = torch.zeros(num_subgraph).to(label.device)  # S
    const = const.scatter_add(0, subgraph_idx,
                              torch.ones_like(subgraph_idx).float())

    reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)

    reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
    reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
    reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

    log_prob = -reduce_adj_loss + reduce_log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1)  # S, K

    bc_log_prob = torch.zeros([B * C]).to(label.device)  # B*C
    bc_idx = torch.arange(B * C).to(label.device)  # B*C
    bc_const = torch.zeros(B * C).to(label.device)
    bc_size = (subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // C  # B
    bc_size = torch.repeat_interleave(bc_size, C)  # B*C
    bc_idx = torch.repeat_interleave(bc_idx, bc_size)  # S
    bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)
    # loss must be normalized for numerical stability
    bc_const = bc_const.scatter_add(0, bc_idx, const)
    bc_loss = (bc_log_prob / bc_const)

    bc_log_prob = bc_log_prob.reshape(B, C)
    bc_loss = bc_loss.reshape(B, C)
    if sum_order_log_prob:
        b_log_prob = torch.sum(bc_log_prob, dim=1)
        b_loss = torch.sum(bc_loss, dim=1)
    else:
        b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
        b_loss = torch.logsumexp(bc_loss, dim=1)

    # probability calculation was for lower-triangular edges
    # must be squared to get probability for entire graph
    b_neg_log_prob = -2 * b_log_prob
    b_loss = -b_loss

    if reduction == "mean":
        neg_log_prob = b_neg_log_prob.mean()
        loss = b_loss.mean()
    elif reduction == "sum":
        neg_log_prob = b_neg_log_prob.sum()
        loss = b_loss.sum()
    else:
        assert reduction == "none"
        neg_log_prob = b_neg_log_prob
        loss = b_loss

    if return_neg_log_prob:
        return loss, neg_log_prob
    else:
        return loss



# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G


def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list, is_real=True):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs


class GRANData(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_sample_subgraph = config.dataset.is_sample_subgraph
    self.num_subgraph_batch = config.dataset.num_subgraph_batch
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute

    if self.is_sample_subgraph:
      assert self.num_subgraph_batch > 0

    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_{}_{}_{}_{}_precompute'.format(
            config.model.name, config.dataset.name, tag, self.block_size,
            self.stride, self.num_canonical_order, self.node_order))

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)

      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        data = self._get_graph_data(G)
        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

  def _get_graph_data(self, G):
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### BFS & DFS from largest-degree node
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)

      bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

    ### k-core
    num_core = nx.core_number(G)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(G.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
      core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
      sort_node_tuple = sorted(
          [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
          key=lambda tt: tt[1],
          reverse=True)
      node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_matrix(G, nodelist=node_list))

    if self.num_canonical_order == 5:
      adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
    else:
      if self.node_order == 'degree_decent':
        adj_list = [adj_1]
      elif self.node_order == 'degree_accent':
        adj_list = [adj_2]
      elif self.node_order == 'BFS':
        adj_list = [adj_3]
      elif self.node_order == 'DFS':
        adj_list = [adj_4]
      elif self.node_order == 'k_core':
        adj_list = [adj_5]
      elif self.node_order == 'DFS+BFS':
        adj_list = [adj_4, adj_3]
      elif self.node_order == 'DFS+BFS+k_core':
        adj_list = [adj_4, adj_3, adj_5]
      elif self.node_order == 'DFS+BFS+k_core+degree_decent':
        adj_list = [adj_4, adj_3, adj_5, adj_1]
      elif self.node_order == 'all':
        adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
      else:
        adj_list = [adj_0]

    # print('number of nodes = {}'.format(adj_0.shape[0]))

    return adj_list

  def __getitem__(self, index):
    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    # load graph
    adj_list = pickle.load(open(self.file_names[index], 'rb'))
    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]
        # adj_tril = np.tril(adj_full, k=-1)

        idx = -1
        for jj in range(0, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
          if jj + K > num_nodes:
            break

          if idx not in rand_idx:
            continue

          ### get graph for GNN propagation
          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=1.0)  # assuming fully connected for the new block
          adj_block = np.tril(adj_block, k=-1)
          adj_block = adj_block + adj_block.transpose()
          adj_block = torch.from_numpy(adj_block).to_sparse()
          edges += [adj_block.coalesce().indices().long()]

          ### get attention index
          # exist node: 0
          # newly added node: 1, ..., K
          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]

          ### get node feature index for GNN input
          # use inf to indicate the newly added nodes where input feature is zero
          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]

          ### get node index for GNN output
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)
          node_idx_gnn += [
              np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)
          ]

          ### get predict label
          label += [
              adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
          ]

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1

      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] = edges[ii] + cum_size[ii]
        node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

      ### pack tensors
      data = {}
      data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
      data['edges'] = torch.cat(edges, dim=1).t().long()
      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
      data['node_idx_feat'] = np.concatenate(node_idx_feat)
      data['label'] = np.concatenate(label)
      data['att_idx'] = np.concatenate(att_idx)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_nodes
      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data_batch += [data]

    end_time = time.time()

    return data_batch

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      data['subgraph_idx_base'] = torch.from_numpy(
        subgraph_idx_base)

      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)

      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0).long()

      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()

      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long()

      # shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1
      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)
      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()

      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float()

      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      batch_data += [data]

    end_time = time.time()
    # print('collate time = {}'.format(end_time - start_time))

    return batch_data


