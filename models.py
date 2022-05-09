import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,adj,mask):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj=adj
        self.nheads=nheads
        self.W = nn.Parameter(torch.empty(size=(nfeat, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attentions = nn.ModuleList([GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.x=nn.Parameter(torch.ones(size=(adj.shape[0],nfeat))/nfeat)
        self.mask=mask.unsqueeze(1).repeat(1,nfeat)
    def forward(self, x,  mode=0):
        if mode==1:
            x=torch.mul(self.x,self.mask)+torch.mul(x,~self.mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if mode == 1:
            return x
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)
