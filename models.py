import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl


class GAT(nn.Module):
    def __init__(self, nfeat,nhid,adj, nclass):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        # self.W=nn.Parameter(pkl.load(open('chameleon_embed.pkl','rb')))
        self.W=nn.Parameter(torch.empty(nfeat,nhid))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.adj=adj
        self.fc=nn.Linear( nhid,nclass)
    def forward(self, x):
        x=torch.mm(x,self.W)
        x=torch.mm(self.adj,x)
        x=F.dropout(x,0.5,self.training)
        x = F.elu(self.fc(x))
        return F.log_softmax(x, dim=1)
