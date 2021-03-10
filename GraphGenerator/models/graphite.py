import torch, math
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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


class GraphiteLayer(Module):
    """
    Simple Graphite layer, similar to https://arxiv.org/abs/1803.10459
    """
    def __init__(self, input_dim, output_dim, bias=True, act=lambda x: x):
        super(GraphiteLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.act = act
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, input1, input2):
        x = torch.mm(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.mm(input1, torch.mm(input1.T, x))+torch.mm(input2, torch.mm(input2.T, x))
        return self.act(x)


class GraphiteVAE(nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim, decode_dim, act=F.relu, autoregressive_scalar=0.5):
        super(GraphiteVAE, self).__init__()
        self.hidden = GraphConvolution(num_features, hidden_dim, act=act)
        self.z_mean = GraphConvolution(hidden_dim, embed_dim, act=act)
        self.mean = None
        self.z_logv = GraphConvolution(hidden_dim, embed_dim, act=act)
        self.logv = None
        self.decode0 = GraphiteLayer(num_features, decode_dim, act=act)
        self.decode1 = GraphiteLayer(embed_dim, decode_dim, act=act)
        self.decode2 = GraphiteLayer(decode_dim, embed_dim, act=lambda x: x)
        self.autoregressive_scalar = autoregressive_scalar

    def forward(self, adj, x=None, device='cuda:0'):
        support = self.hidden(x, adj)
        self.mean = self.z_mean(support, adj)
        self.logv = self.z_logv(support, adj)
        noise = Variable(torch.rand(self.mean.shape[0], self.mean.shape[1], dtype=torch.float32)).to(device)
        support = noise * torch.exp(self.logv) + self.mean
        recon_1 = F.normalize(support, p=2, dim=1)
        recon_2 = torch.ones(recon_1.shape).to(device)
        recon_2 /= torch.sqrt(recon_2.sum(1, keepdim=True))
        d = torch.mm(recon_1, torch.unsqueeze(recon_1.sum(0), 1)) + \
            torch.mm(recon_2, torch.unsqueeze(recon_2.sum(0), 1))
        d = d.pow(-0.5)
        recon_1 = recon_1*d
        recon_2 = recon_2*d
        update = self.decode1(support, recon_1, recon_2) + self.decode0(x, recon_1, recon_2)
        update = self.decode2((update, recon_1, recon_2))
        update = (1-self.autoregressive_scalar) * support + self.autoregressive_scalar * update
        reconstructions = torch.mm(update, update.T)
        return reconstructions
        # return update


class GraphiteAE(nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim, decode_dim, act=F.relu, autoregressive_scalar=0.5):
        super(GraphiteAE, self).__init__()
        self.hidden = GraphConvolution(num_features, hidden_dim, act=act)
        self.z_mean = GraphConvolution(hidden_dim, embed_dim, act=act)
        self.mean = None
        # self.z_logv = GraphConvolution(hidden_dim, embed_dim, act=act)
        # self.logv = None
        self.decode0 = GraphiteLayer(num_features, decode_dim, act=act)
        self.decode1 = GraphiteLayer(embed_dim, decode_dim, act=act)
        self.decode2 = GraphiteLayer(decode_dim, embed_dim, act=lambda x: x)
        self.autoregressive_scalar = autoregressive_scalar

    def forward(self, adj, x=None, device='cuda:0'):
        support = self.hidden(x, adj)
        support = self.z_mean(support, adj)
        # self.logv = self.z_logv(support, adj)
        # noise = Variable(torch.rand(self.mean.shape[0], self.mean.shape[1], dtype=torch.float32)).to(device)
        # support = noise * torch.exp(self.logv) + self.mean
        recon_1 = F.normalize(support, p=2, dim=1)
        recon_2 = torch.ones(recon_1.shape).to(device)
        recon_2 /= torch.sqrt(recon_2.sum(1, keepdim=True))
        d = torch.mm(recon_1, torch.unsqueeze(recon_1.sum(0), 1)) + \
            torch.mm(recon_2, torch.unsqueeze(recon_2.sum(0), 1))
        d = d.pow(-0.5)
        recon_1 = recon_1 * d
        recon_2 = recon_2 * d
        update = self.decode1(support, recon_1, recon_2) + self.decode0(x, recon_1, recon_2)
        update = self.decode2(update, recon_1, recon_2)
        update = (1 - self.autoregressive_scalar) * support + self.autoregressive_scalar * update
        reconstructions = torch.mm(update, update.T)
        return reconstructions
        # return update

