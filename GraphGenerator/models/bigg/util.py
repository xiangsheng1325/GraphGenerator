# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from bigg.common.consts import t_float
t_float = torch.float32
# from bigg.common.pytorch_util import glorot_uniform, MLP


def _glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        _glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        _glorot_uniform(m.weight.data)
    elif isinstance(m, nn.Embedding):
        _glorot_uniform(m.weight.data)


def glorot_uniform(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList) or isinstance(p, nn.ModuleList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
        super(MLP, self).__init__()
        self.act_last = act_last
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.bn = bn

        if isinstance(hidden_dims, str):
            hidden_dims = list(map(int, hidden_dims.split("-")))
        assert len(hidden_dims)
        hidden_dims = [input_dim] + hidden_dims
        self.output_size = hidden_dims[-1]

        list_layers = []

        for i in range(1, len(hidden_dims)):
            list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i + 1 < len(hidden_dims):  # not the last layer
                if self.bn:
                    bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
                    list_layers.append(bnorm_layer)
                list_layers.append(NONLINEARITIES[self.nonlinearity])
                if dropout > 0:
                    list_layers.append(nn.Dropout(dropout))
            else:
                if act_last is not None:
                    list_layers.append(NONLINEARITIES[act_last])

        self.main = nn.Sequential(*list_layers)

    def forward(self, z):
        x = self.main(z)
        return x


class TreeLSTMCell(nn.Module):
    def __init__(self, arity, latent_dim):
        super(TreeLSTMCell, self).__init__()
        self.arity = arity
        self.latent_dim = latent_dim

        self.mlp_i = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='sigmoid')

        self.mlp_o = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='sigmoid')

        self.mlp_u = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='tanh')

        f_list = []
        for _ in range(arity):
            mlp_f = MLP(arity * latent_dim, [2 * arity * latent_dim, latent_dim], act_last='tanh')
            f_list.append(mlp_f)
        self.f_list = nn.ModuleList(f_list)

    def forward(self, list_h_mat, list_c_mat):
        assert len(list_c_mat) == self.arity == len(list_h_mat)
        h_mat = torch.cat(list_h_mat, dim=-1)
        assert h_mat.shape[1] == self.arity * self.latent_dim

        i_j = self.mlp_i(h_mat)

        f_sum = 0
        for i in range(self.arity):
            f = self.f_list[i](h_mat)
            f_sum = f_sum + f * list_c_mat[i]

        o_j = self.mlp_o(h_mat)

        u_j = self.mlp_u(h_mat)

        c_j = i_j * u_j + f_sum

        h_j = o_j * torch.tanh(c_j)

        return h_j, c_j


class BinaryTreeLSTMCell(TreeLSTMCell):
    def __init__(self, latent_dim):
        super(BinaryTreeLSTMCell, self).__init__(2, latent_dim)

    def forward(self, lch_state, rch_state):
        list_h_mat, list_c_mat = zip(lch_state, rch_state)
        return super(BinaryTreeLSTMCell, self).forward(list_h_mat, list_c_mat)


class AdjNode(object):
    def __init__(self, parent, row, col_range, depth):
        self.parent = parent
        self.depth = depth
        self.row = row
        self.col_range = col_range

        self.n_cols = self.col_range[1] - self.col_range[0]
        self.is_leaf = self.n_cols <= 1
        self.is_root = self.parent is None
        self.edge = None
        self.has_edge = False
        if self.is_leaf:
            self.edge = (self.row, self.col_range[0])

        self._state = None
        self._bits_rep = None
        self.lch = None
        self.rch = None

    @property
    def state(self):
        return self._state

    @property
    def bits_rep(self):
        if self._bits_rep is None and not self.is_leaf:
            bits = []
            if self.lch._bits_rep is not None:
                offset = self.rch.col_range[1] - self.rch.col_range[0]
                bits += [x + offset for x in self.lch._bits_rep]
            if self.rch._bits_rep is not None:
                bits += self.rch._bits_rep
            self._bits_rep = bits
        return self._bits_rep

    @bits_rep.setter
    def bits_rep(self, new_bits):
        self._bits_rep = new_bits

    @state.setter
    def state(self, new_state):
        self._state = new_state

    def split(self):
        if (self.lch is not None and self.rch is not None) or self.is_leaf:
            return

        col_ranges = [(self.col_range[0], self.col_range[0] + self.n_cols // 2),
                        (self.col_range[0] + self.n_cols // 2, self.col_range[1])]
        self.lch = AdjNode(self, self.row, col_ranges[0], self.depth + 1)
        self.rch = AdjNode(self, self.row, col_ranges[1], self.depth + 1)


class ColAutomata(object):
    def __init__(self, supervised, indices=None):
        self.pos = 0
        self.indices = indices
        self.supervised = supervised
        if indices is None:
            self.indices = []

    @property
    def next_edge(self):
        if self.pos < len(self.indices):
            return self.indices[self.pos]
        else:
            return None

    @property
    def last_edge(self):
        if self.pos < len(self.indices):
            return self.indices[-1]
        else:
            return None

    def add_edge(self, col_idx):
        self.pos += 1
        if not self.supervised:
            self.indices.append(col_idx)

    def has_edge(self, range_start, range_end):
        for i in self.indices:
            if i >= range_start and i < range_end:
                return True
        return False


class AdjRow(object):
    def __init__(self, row, directed=False, self_loop=False, col_range=None):
        self.row = row
        assert not directed
        max_col = row
        if self_loop:
            max_col += 1
        if col_range is None:
            col_range = (0, max_col)
        self.root = AdjNode(None, row, col_range, 0)