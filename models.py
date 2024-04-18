import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN_plus(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer):
        super(GCN_plus, self).__init__()

        self.gc_layers_A1 = nn.ModuleList([GraphConvolution(nfeat if i == 0 else nhid, nhid) for i in range(nlayer)])
        self.gc_layers_A2 = nn.ModuleList([GraphConvolution(nfeat if i == 0 else nhid, nhid) for i in range(nlayer)])
        self.gc_layers_P1 = nn.ModuleList([GraphConvolution(nfeat if i == 0 else nhid, nhid) for i in range(nlayer)])
        self.gc_layers_P2 = nn.ModuleList([GraphConvolution(nfeat if i == 0 else nhid, nhid) for i in range(nlayer)])

        self.Linear_layer_A1 = nn.Linear(in_features=nhid, out_features=nclass)
        self.Linear_layer_A2 = nn.Linear(in_features=nhid, out_features=nclass)
        self.Linear_layer_P1 = nn.Linear(in_features=nhid, out_features=nclass)
        self.Linear_layer_P2 = nn.Linear(in_features=nhid, out_features=nclass)

        self.Linear_layer_fusion = nn.Linear(in_features=nclass * 2, out_features=nclass)

        self.dropout = dropout

    def forward(self, x, A1, P1, A2, P2):
        x_A1 = x
        for conv in self.gc_layers_A1:
            x_A1 = torch.tanh(conv(x_A1, A1))
            x_A1 = F.dropout(x_A1, self.dropout, training=self.training)
        x_A1 = self.Linear_layer_A1(x_A1)

        x_P1 = x
        for conv in self.gc_layers_P1:
            x_P1 = torch.tanh(conv(x_P1, P1))
            x_P1 = F.dropout(x_P1, self.dropout, training=self.training)
        x_P1 = self.Linear_layer_P1(x_P1)

        x_A2 = x
        for conv in self.gc_layers_A2:
            x_A2 = torch.tanh(conv(x_A2, A2))
            x_A2 = F.dropout(x_A2, self.dropout, training=self.training)
        x_A2 = self.Linear_layer_A2(x_A2)

        x_P2 = x
        for conv in self.gc_layers_P2:
            x_P2 = torch.tanh(conv(x_P2, P2))
            x_P2 = F.dropout(x_P2, self.dropout, training=self.training)
        x_P2 = self.Linear_layer_P2(x_P2)

        x = torch.cat((x_A1, x_A2), 1)
        x = self.Linear_layer_fusion(x)

        return F.log_softmax(x, dim=1), F.log_softmax(x_P1, dim=1), F.log_softmax(x_P2, dim=1), x_A1
