#!/usr/bin/env python
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

from utils.utils import sparse_mx_to_torch_sparse_tensor


class Sampler:
    def __init__(self, features, adj, **kwargs):
        allowed_kwargs = {'input_dim', 'layer_sizes', 'device'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, \
                'Invalid keyword argument: ' + kwarg

        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')
        self.device = kwargs.get('device', torch.device("cpu"))

        self.num_layers = len(self.layer_sizes)

        self.adj = adj
        self.features = features

        self.train_nodes_number = self.adj.shape[0]

    def sampling(self, v_indices):
        raise NotImplementedError("sampling is not implemented")

    def _change_sparse_to_tensor(self, adjs):
        new_adjs = []
        for adj in adjs:
            new_adjs.append(
                sparse_mx_to_torch_sparse_tensor(adj).to(self.device))
        return new_adjs


class SamplerFastGCN(Sampler):
    def __init__(self, pre_probs, features, adj, **kwargs):
        super(SamplerFastGCN, self).__init__(features, adj, **kwargs)
        col_norm = sparse_norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, v):
        all_support = [[]] * self.num_layers

        cur_out_nodes = v
        for layer_index in range(self.num_layers - 1, -1, -1):
            cur_sampled, cur_support = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_support[layer_index] = cur_support
            cur_out_nodes = cur_sampled

        all_support = self._change_sparse_to_tensor(all_support)
        sampled_X0 = self.features[cur_out_nodes]
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size):
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1)

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        self.embedding = output
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class FastGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super(FastGCN, self).__init__()

        self.layer1 = GraphConvolution(nfeat, nhid)
        self.layer2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        outputs1 = F.relu(self.layer1(x, adj[0]))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.layer2(outputs1, adj[1])
        return F.log_softmax(outputs2, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)
