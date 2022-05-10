import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
 
 
class word2vec():
 
    def __init__(self,nhid,lr=0.01,epochs=100):
        self.lr =lr
        self.epochs = epochs
        self.node_number=-1
        self.vector_number=-1
        self.nhid=nhid
    def generate_training_data(self, adj,features):
        """
        得到训练数据
        """
 
        training_data = []

        for i in range(self.node_number):
            w_target = features[i]
            w_context = []
            for j in torch.nonzero(adj[0]):
                w_context.append(features[j])
            training_data.append([w_target, w_context])
 
        return (training_data)
 
    def train(self, training_data):
 
 
        #随机化参数w1,w2
        self.w1 = torch.randn((self.vector_number,self.nhid))#
 
        self.w2 = torch.randn((self.nhid, self.vector_number))

 
        for i in range(self.epochs):
 
            self.loss = 0
 
            for w_t, w_c in training_data:
 
                #前向传播
                y_pred, h, u = self.forward(w_t)
 
                #计算误差
                EI = sum([(y_pred- word) for word in w_c])
 
                #反向传播，更新参数
                self.backprop(EI, h, w_t)
 
                #计算总损失
            #     self.loss += -torch.sum([u[word.index(1)] for word in w_c]) + len(w_c) * torch.log(torch.sum(torch.exp(u)))
 
            print('Epoch:', i)
 
    def forward(self, x):
        """
        前向传播
        """
        x=(x.unsqueeze(0))
        h = torch.mm(x,self.w1)#1,hid
 
        u = torch.mm(h, self.w2).flatten()#1,v
 
        y_c = F.softmax(u,dim=-1)
 
        return y_c, h, u
 
 
    def softmax(self, x):
        """
        """
        e_x = torch.exp(x - torch.max(x))
 
        return e_x / torch.sum(e_x)
 
 
    def backprop(self, e, h, x):
        d1_dw2 = torch.outer(h.flatten(), e.flatten())
        d1_dw1 = torch.outer(x.flatten(), torch.mm(self.w2, e.T).flatten())
 
        self.w1 = self.w1 - (self.lr * d1_dw1)
        self.w2 = self.w2 - (self.lr * d1_dw2)
    
    def __call__(self,adj,features):
        self.node_number,self.vector_number=features.shape
        training_data = self.generate_training_data(adj,features)
        self.train(training_data)
        return self.w1
from process import full_load_data
dataset='chameleon'
datastr = dataset
splitstr = 'splits/'+dataset+'_split_0.6_0.2_0.npz'
adj, adj_i, features, labels, idx_train, idx_val, idx_test = full_load_data(
        datastr, splitstr)
adj=adj.to_dense()
import pickle as pkl
w2v=word2vec(8)
res=(w2v(adj,features))
pkl.dump(res,open(dataset+'_embed.pkl','wb'))