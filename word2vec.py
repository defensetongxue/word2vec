import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torch.optim as optim
import pickle as pkl
from process import full_load_data
 
class Word2Vec(nn.Module):
  def __init__(self,voc_size,nhid):
    super(Word2Vec, self).__init__()

    # W and V is not Traspose relationship
    self.W = nn.Parameter(torch.randn(voc_size, nhid))
    self.V = nn.Parameter(torch.randn(nhid, voc_size))

  def forward(self, X):
    # X : [batch_size, voc_size] 
    # torch.mm only for 2 dim matrix, but torch.matmul can use to any dim
    hidden_layer = torch.matmul(X, self.W) # hidden_layer : [batch_size, embedding_size]
    output_layer = torch.matmul(hidden_layer, self.V) # output_layer : [batch_size, voc_size]
    return output_layer
class build_embed():
    def __init__(self,features,adj,nhid,lr=0.01,epoch=2000) :
        node_number,vector_size=features.shape
        self.node_number=node_number
        self.model=Word2Vec(vector_size,nhid)
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epoch=epoch
        self.data_loader=self._build_data(features,adj)
    def _build_data(self,features,adj):
        batch=[]
        X,Y=[],[]
        edge_list=torch.nonzero(adj)
        cnt=1000
        for i,j in edge_list:
            X.append(features[i].unsqueeze(0))
            Y.append(features[j].unsqueeze(0))
            cnt-=1
            if cnt==0:
                cnt=1000
                batch.append([torch.cat(X,dim=0),torch.cat(Y,dim=0)])
                X,Y=[],[]
        adj=torch.mm(adj,adj)
        edge_list=torch.nonzero(adj)
        for i,j in edge_list:
            X.append(features[i].unsqueeze(0))
            Y.append(features[j].unsqueeze(0))
            cnt-=1
            if cnt==0:
                cnt=1000
                batch.append([torch.cat(X,dim=0),torch.cat(Y,dim=0)])
                X,Y=[],[]
        batch.append([torch.cat(X,dim=0),torch.cat(Y,dim=0)])
        self.batch_number=len(batch)
        print('build pretrain data finished, totally {} batches, each batch has 1000 data'.format(self.batch_number))
        return batch
    def train(self):
        print('begin the pretrain process')
        
        for epoch in range(self.epoch):
            epoch_loss=0
            for data in self.data_loader:
                pred = self.model(data[0])
                loss = self.criterion(pred, data[1])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss+=loss.item()
            if (epoch + 1) % 100 == 0:
                  print(epoch + 1,  epoch_loss/self.batch_number)
        return 
    def get_result(self,dataset):
        W=self.model._parameters['W'].detach()
        tmp_file=open(dataset+'_embed.pkl','wb')
        pkl.dump(W,tmp_file)
        tmp_file.close()
        return 
dataset='cora'
datastr = dataset
splitstr = 'splits/'+dataset+'_split_0.6_0.2_0.npz'
adj, adj_i, features, labels, idx_train, idx_val, idx_test = full_load_data(
        datastr, splitstr)
adj=adj.to_dense()
model=build_embed(features,adj,8)
model.train()
model.get_result(dataset)
