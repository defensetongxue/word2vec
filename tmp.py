from process import full_load_data
from word2vec import word2vec
dataset='chameleon'
datastr = dataset
splitstr = 'splits/'+dataset+'_split_0.6_0.2_0.npz'
adj, adj_i, features, labels, idx_train, idx_val, idx_test = full_load_data(
        datastr, splitstr)

adj=adj.to_dense()