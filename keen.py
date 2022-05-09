import torch
import networkx as nx
import random
def adj2nx(adj):
    G = nx.Graph()
    n=adj.shape[0]
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i,n):
            G.add_edge(i,j)
    return nx.adjacency_matrix(G)
def generate_mask(adj,mode='degree'):
    if mode=='degree':
        degree=torch.zeros(size=(1,adj.shape[0])).squeeze(0)
        for i in range(adj.shape[0]):
            degree[i]=torch.nonzero(adj[i]).shape[0]
        # total_train=torch.sum(idx_train)
        res=torch.ones(size=(1,adj.shape[0])).squeeze(0)
        print("generate mask data ,totally have pretrain data {}".format(sum(res)))
        return torch.tensor(res,dtype=torch.bool)
    elif mode=='random':
        n=adj.shape[0]
        res=torch.zeros(size=(1,n)).squeeze(0)
        chosed=torch.zeros(size=(1,n)).squeeze(0)
        cnt=0
        patient=0
        while(True):
            c=random.randint(0,n-1)
            if chosed[c]==0:
                res[c]=1
                chosed+=adj[c]
                chosed[c]=1
                cnt+=1
                patient=0
            else:
                patient+=1
            if patient==100:
                print("rate is too high")
                break
        print("generate mask data ,totally have pretrain data {}".format(sum(res)))
        return torch.tensor(res,dtype=torch.bool)