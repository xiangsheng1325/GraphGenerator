import torch, math
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.modules import ModuleList
from torch.nn.parameter import Parameter
import torch.nn.functional as F
SMALL = 1e-16
EULER_GAMMA = 0.5772156649015329


def reparametrize_discrete(logalphas, temp, shape=None):
    """
    input: logit, output: logit
    """
    if shape is None:
        shape = logalphas.shape
    uniform = torch.Tensor(shape).uniform_(1e-4, 1. - 1e-4)
    logistic = torch.log(uniform) - torch.log(1. - uniform)
    ysample = (logalphas + logistic) / temp
    return ysample


def sample(z_mean, z_log_std, pi_logit, a, b, temp, calc_v=True, calc_real=True):
    if calc_real:
        # mu + standard_samples * stand_deviation
        z_real = z_mean + torch.Tensor(z_mean.shape).uniform_(1e-4, 1. - 1e-4) * torch.exp(z_log_std)
    else:
        z_real = None

    # Concrete instead of Bernoulli
    y_sample = reparametrize_discrete(pi_logit, temp)
    z_discrete = F.sigmoid(y_sample)

    if calc_v:
        # draw v from kumarswamy instead of Beta
        v = kumaraswamy_sample(a, b)
    else:
        v = None

    return z_discrete, z_real, v, y_sample


def logit(x):
    return torch.log(x+SMALL) - torch.log(1. - x + SMALL)


def kumaraswamy_sample(a, b):
    u = torch.Tensor(a.shape).uniform_(1e-4, 1. - 1e-4)
    # return (1. - u.pow(1./b)).pow(1./a)
    return torch.exp(torch.log(1. - torch.exp(torch.log(u) / (b+SMALL)) + SMALL) / (a+SMALL))


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

    def forward(self, input, adj, training=True):
        support = torch.dropout(input, p=self.dropout, train=training)
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
    def __init__(self, input_dim, hidden_dim=None, num_classes=0, dropout=0.5, config=None):
        super(SBMGNN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.config = config
        self.hidden = [int(x) for x in hidden_dim]
        self.num_layers = len(self.hidden)
        self.h = GraphConvolution(in_features=input_dim,
                                  out_features=self.hidden[0],
                                  act=nn.LeakyReLU(negative_slope=0.2),
                                  dropout=self.dropout)
        h_mid = []
        for i in range(self.num_layers-2):
            h_mid.append(GraphConvolution(in_features=self.hidden[i],
                                          out_features=self.hidden[i+1],
                                          act=nn.LeakyReLU(negative_slope=0.2),
                                          dropout=self.dropout))
        self.h_mid = ModuleList(h_mid)
        self.h1 = GraphConvolution(in_features=self.hidden[-2],
                                   out_features=self.hidden[-1],
                                   act=lambda x: x,
                                   dropout=self.dropout)
        self.h2 = GraphConvolution(in_features=self.hidden[-2],
                                   out_features=self.hidden[-1],
                                   act=lambda x: x,
                                   dropout=self.dropout)
        self.h3 = GraphConvolution(in_features=self.hidden[-2],
                                   out_features=self.hidden[-1],
                                   act=lambda x: x,
                                   dropout=self.dropout)
        self.deep_decoder = nn.Sequential(nn.Linear(self.hidden[self.num_layers-1], config.model.g_hidden),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Linear(config.model.g_hidden, config.model.g_hidden//2))
        self.mean = None
        self.logv = None
        self.pi_logit = None
        self.a = None
        self.beta_a = None
        self.b = None
        self.beta_b = None
        self.v = None
        self.x_hat = None
        self.logit_post = None
        self.log_prior = None
        self.z_discrete = None
        self.z_real = None
        self.y_sample = None
        self.reconstructions = None
        self.get_alpha_beta(config=self.config)

    def get_alpha_beta(self, config, training=False):
        a_val = np.log(np.exp(config.model.alpha0) - 1)
        b_val = np.log(np.exp(1.) - 1)
        initial = torch.zeros(self.hidden[self.num_layers-1], dtype=torch.float32).to(config.device)
        self.a = Variable(initial) + a_val
        self.b = Variable(initial) + b_val
        beta_a = F.softplus(self.a)
        beta_b = F.softplus(self.b)
        beta_a = torch.unsqueeze(beta_a, 0)
        beta_b = torch.unsqueeze(beta_b, 0)
        self.beta_a = torch.tile(beta_a, [config.model.num_nodes, 1])
        self.beta_a = Variable(self.beta_a, requires_grad=True)
        self.beta_b = torch.tile(beta_b, [config.model.num_nodes, 1])
        self.beta_b = Variable(self.beta_b, requires_grad=True)

    def get_reconstructions(self, config, training):
        self.v = kumaraswamy_sample(self.beta_a, self.beta_b)
        v_term = torch.log(self.v + SMALL)
        self.log_prior = torch.cumsum(v_term, axis=1)
        self.logit_post = self.pi_logit + logit(torch.exp(self.log_prior))
        self.z_discrete, self.z_real, _, self.y_sample = sample(self.mean,
                                                                self.logv,
                                                                self.logit_post,
                                                                None, None,
                                                                config.model.temp_post,
                                                                calc_v=False)
        self.z_discrete = torch.round(self.z_discrete) if not training else self.z_discrete
        z = torch.mul(self.z_discrete, self.z_real)
        if config.model.deep_decoder:
            z = self.deep_decoder(z)
        self.reconstructions = torch.mm(z, torch.transpose(z, 1, 0))
        return self.reconstructions

    def forward(self, adj, x=None, device='cuda:0', training=True):
        support = self.h(x, adj, training=training)
        for h in self.h_mid:
            support = h(support, adj, training=training)
        self.mean = self.h1(support, adj, training=training)
        self.logv = self.h2(support, adj, training=training)
        self.pi_logit = self.h3(support, adj, training=training)
        return self.get_reconstructions(config=self.config, training=training)

    def monte_carlo_sample(self, pi_logit, z_mean, z_log_std, temp, S, sigmoid_fn):
        shape = list(pi_logit.shape)
        shape.insert(0, S)
        y_sample = reparametrize_discrete(pi_logit, temp, shape)
        z_discrete = torch.sigmoid(y_sample)
        z_discrete = torch.round(z_discrete)
        noise = Variable(torch.rand(z_mean.shape[0], z_mean.shape[1], dtype=torch.float32)).to(self.config.device)
        z_real = noise * torch.exp(z_log_std) + z_mean
        emb = torch.mul(z_real, z_discrete)
        if self.config.model.deep_decoder:
            emb = self.deep_decoder(emb)
        emb_t = emb.permute((0, 2, 1))
        adj_rec = torch.matmul(emb, emb_t)
        adj_rec = adj_rec.mean(dim=0)
        z_activated = torch.sum(z_discrete) / (shape[0] * shape[1])
        return adj_rec, z_activated


def train_sbmgnn(sp_adj, feature, config=None):
    print('Using gpus: ' + str(config.gpu))
    print('---------------------------------')
    print('Dataset: ' + config.dataset.name + '_' + str(config.train.split_idx))
    print('Alpha0: ' + str(config.model.alpha0))
    print('WeightedCE: ' + str(config.train.weighted_ce))
    print('ReconstructX: ' + str(config.train.reconstruct_x))
    model = SBMGNN(input_dim=config.model.num_nodes,
                   hidden_dim=config.model.hidden,
                   config=conf)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    pass


if __name__ == '__main__':
    class tmpmodl:
        g_hidden = 16
        deep_decoder = 1
        temp_post = 1.
        alpha0 = 10.
        num_nodes = 2

    class tmpconf:
        def __init__(self):
            self.input_dim = 16
            self.hidden_dim = [32, 16]
            self.model = tmpmodl
            self.device = 'cpu'
            self.lr = 0.01

    conf = tmpconf()
    model = SBMGNN(input_dim=conf.input_dim,
                   hidden_dim=conf.hidden_dim,
                   config=conf)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    x = torch.ones(conf.model.num_nodes, conf.input_dim)
    adj = torch.ones(conf.model.num_nodes, conf.model.num_nodes)
    tmp = model(adj, x, device=conf.device)
    loss = F.binary_cross_entropy_with_logits(tmp, adj)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tmp1 = model(adj, x, device=conf.device)
