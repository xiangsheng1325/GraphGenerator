import torch, math
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, act=lambda x: x):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = act
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
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VGAE(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, act=lambda x: x, layers=2):
        super(VGAE, self).__init__()
        self.encode = GraphConvolution(input_size, hidden_size, act=act)
        self.medium = nn.ModuleList([GraphConvolution(hidden_size, hidden_size, act=act) for i in range(layers-2)])
        self._mean = GraphConvolution(hidden_size, emb_size, act=act)
        self._logv = GraphConvolution(hidden_size, emb_size, act=act)
        self.mean = None
        self.logv = None

    def forward(self, adj, x=None, device='cuda:0'):
        if x is None:
            x = Variable(torch.rand(adj.shape[0], self.input_size, dtype=torch.float32)).to(device)
        support = self.encode(x, adj)
        for m in self.medium:
            support = m(support, adj)
        self.mean = self._mean(support, adj)
        self.logv = self._logv(support, adj)
        noise = Variable(torch.rand(self.mean.shape[0], self.mean.shape[1], dtype=torch.float32)).to(device)
        support = noise*torch.exp(self.logv) + self.mean
        score = torch.mm(support, support.T)
        return score


class GAE(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, act=lambda x: x, layers=2):
        super(GAE, self).__init__()
        self.encode = GraphConvolution(input_size, hidden_size, act=act)
        self.medium = nn.ModuleList([GraphConvolution(hidden_size, hidden_size, act=act) for i in range(layers-2)])
        self.mean = GraphConvolution(hidden_size, emb_size, act=act)

    def forward(self, adj, x=None, device='cuda:0'):
        if x is None:
            x = Variable(torch.rand(adj.shape[0], self.input_size, dtype=torch.float32)).to(device)
        support = self.encode(x, adj)
        for m in self.medium:
            support = m(support, adj)
        support = self.mean(support, adj)
        score = torch.mm(support, support.T)
        return score


