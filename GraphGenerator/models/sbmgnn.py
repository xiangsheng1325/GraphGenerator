import torch, math
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.modules import ModuleList
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, act=lambda x: x, dropout=0.5):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = act
        self.dropout = dropout
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.dropout(input, p=self.dropout)
        support = torch.mm(support, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SBMGNN(Module):
    def __init__(self, input_dim, hidden_dim=None, num_classes=0, dropout=0.5):
        super(SBMGNN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden = [int(x) for x in hidden_dim]
        self.num_layers = len(self.hidden)
        h = GraphConvolution(in_features=input_dim,
                             out_features=self.hidden[0],
                             act=nn.LeakyReLU(negative_slope=0.2),
                             dropout=self.dropout)
        h_mid = []
        for i in range(self.num_layers-2):
            h_mid.append(GraphConvolution(in_features=self.hidden[i],
                             out_features=self.hidden[i+1],
                             act=nn.LeakyReLU(negative_slope=0.2),
                             dropout=self.dropout))
        h_mid = ModuleList(h_mid)
        h1 = GraphConvolution(in_features=self.hidden[-2],
                              out_features=self.hidden[-1],
                              act=lambda x: x,
                              dropout=self.dropout)
        h2 = GraphConvolution(in_features=self.hidden[-2],
                              out_features=self.hidden[-1],
                              act=lambda x: x,
                              dropout=self.dropout)
        h3 = GraphConvolution(in_features=self.hidden[-2],
                              out_features=self.hidden[-1],
                              act=lambda x: x,
                              dropout=self.dropout)

    def forward(self):
        pass
