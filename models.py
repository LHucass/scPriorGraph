import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN, self).__init__()

        self.gcA1_1 = GraphConvolution(nfeat, nhid1)
        self.gcA1_2 = GraphConvolution(nhid1, nhid2)
        self.Linear_layer_A1 = nn.Linear(in_features=nhid2, out_features=nclass)
        self.dropout = dropout

        self.gcA2_1 = GraphConvolution(nfeat, nhid1)
        self.gcA2_2 = GraphConvolution(nhid1, nhid2)
        self.Linear_layer_A2 = nn.Linear(in_features=nhid2, out_features=nclass)

        self.gcP1_1 = GraphConvolution(nfeat, nhid1)
        self.gcP1_2 = GraphConvolution(nhid1, nhid2)
        self.Linear_layer_P1 = nn.Linear(in_features=nhid2, out_features=nclass)

        self.gcP2_1 = GraphConvolution(nfeat, nhid1)
        self.gcP2_2 = GraphConvolution(nhid1, nhid2)
        self.Linear_layer_P2 = nn.Linear(in_features=nhid2, out_features=nclass)

        self.Linear_layer_fusion = nn.Linear(in_features=nclass * 2, out_features=nclass)

    def forward(self, x, A1, P1, A2, P2):
        x_A1 = torch.tanh(self.gcA1_1(x, A1))
        x_A1 = F.dropout(x_A1, self.dropout, training=self.training)
        x_A1 = torch.tanh(self.gcA1_2(x_A1, A1))
        x_A1 = F.dropout(x_A1, self.dropout, training=self.training)
        x_A1 = self.Linear_layer_A1(x_A1)

        x_P1 = torch.tanh(self.gcP1_1(x, P1))
        x_P1 = F.dropout(x_P1, self.dropout, training=self.training)
        x_P1 = torch.tanh(self.gcP1_2(x_P1, P1))
        x_P1 = F.dropout(x_P1, self.dropout, training=self.training)
        x_P1 = self.Linear_layer_P1(x_P1)

        x_A2 = torch.tanh(self.gcA2_1(x, A2))
        x_A2 = F.dropout(x_A2, self.dropout, training=self.training)
        x_A2 = torch.tanh(self.gcA2_2(x_A2, A2))
        x_A2 = F.dropout(x_A2, self.dropout, training=self.training)
        x_A2 = self.Linear_layer_A2(x_A2)

        x_P2 = torch.tanh(self.gcP2_1(x, P2))
        x_P2 = F.dropout(x_P2, self.dropout, training=self.training)
        x_P2 = torch.tanh(self.gcP2_2(x_P2, P2))
        x_P2 = F.dropout(x_P2, self.dropout, training=self.training)
        x_P2 = self.Linear_layer_P2(x_P2)

        x = torch.cat((x_A1, x_A2), 1)
        x = self.Linear_layer_fusion(x)

        return F.log_softmax(x, dim=1), F.log_softmax(x_P1, dim=1), F.log_softmax(x_P2, dim=1), x

