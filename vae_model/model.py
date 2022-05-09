import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden):
        super(GCNModelVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_feat_dim, hidden*12),
            nn.ReLU(True),
            nn.Linear(hidden*12, hidden*4),
            nn.ReLU(True),
            nn.Linear(hidden*4, hidden))

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden*4),
            nn.ReLU(True),
            nn.Linear(hidden*4, hidden*12),
            nn.ReLU(True),
            nn.Linear(hidden*12,input_feat_dim ),
            nn.Tanh())

    def forward(self, x):
        z= self.encoder(x)
        x = self.decoder(z)
        return z,x
