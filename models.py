import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nclass):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.fc=nn.Linear( nfeat,nclass)
    def forward(self, x):
        x = F.elu(self.fc(x))
        return F.log_softmax(x, dim=1)
