from __future__ import division
from __future__ import print_function


import time

import torch
from torch import optim

from vae_model.model import GCNModelVAE

import torch.nn.functional as F
from progress.bar import Bar

class vae():
    def __init__(self, nhid, epochs=6000, lr=0.02, dropout=0.) -> None:
        self.nhid=nhid
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout

    def __call__(self, features):
        print("begin to generate pretrain_data using gcn_vae method")
        n_nodes, feat_dim = features.shape


        encoder_model = GCNModelVAE(
            feat_dim, self.nhid)
        optimizer = optim.Adam(encoder_model.parameters(), lr=self.lr)
        hidden_emb = None
        bar = Bar('gcn_vae', max=self.epochs)
        best=100000
        patient=100
        bad_count=0
        for epoch in range(self.epochs):
            t = time.time()
            encoder_model.train()
            optimizer.zero_grad()
            z,x=encoder_model(features)
            loss = F.mse_loss(features,x)
            loss.backward()
            optimizer.step()
            if loss<best:
                bad_count=0
                hidden_emb=z
            else:
                bad_count+=1
            if bad_count==patient:
                break
            if (epoch+1)%500==0:
                print("epoches {} loss {:.5f}".format(epoch+1,loss))
            bar.next()
        bar.finish()
        if bad_count==patient:
            print("收敛")
        print("generate nhid embeding finished")
        return hidden_emb
