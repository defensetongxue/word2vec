from __future__ import division
from __future__ import print_function
import time
import random
import argparse
from vae_model.vae import vae
from vae_model.pretrain import do_pretrain_job
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from models import *
import uuid
from keen import generate_mask
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')  # Default seed same as GCNII
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhid', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--pretrain', type=int, default=0, help='Patience')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print("==========================")
print(f"Dataset: {args.data}")

checkpt_file = 'pretrained/best.pt'

def train_step(model, optimizer, labels, features, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate_step(model, labels, features, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        return loss_val.item(), acc_val.item()


def test_step(model, labels,features, idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        # print(mask_val)
        return loss_test.item(), acc_test.item()


def train(datastr, splitstr):
    adj, adj_i, features, labels, idx_train, idx_val, idx_test = full_load_data(
        datastr, splitstr)
    adj = adj.to_dense()
    adj_i = adj_i.to_dense()
    mask=generate_mask(adj,'random')
    model = GAT(nfeat=features.shape[1], 
                nhid=args.nhid, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha
                ,adj=adj,
                mask=mask)
    if args.pretrain:
        tokenizer=vae(nhid=args.nhid,epochs=7000,lr=0.02)
        pretrain_class=do_pretrain_job(tokenizer,epochs=2000,nheads=args.nb_heads)
        pretrain_class(model,features,mask)
        model.load_state_dict(torch.load("pretrained/prebest.pt"))
        
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train_step(
            model, optimizer, labels, features, idx_train)
        loss_val, acc_val = validate_step(
            model, labels,features, idx_val)
        # Uncomment following lines to see loss and accuracy values

        if(epoch+1) % 10 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra*100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val*100))

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    test_out = test_step(model, labels,features, idx_test)
    acc = test_out[1]

    return acc*100


t_total = time.time()
acc_list = []

for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    accuracy_data = train(datastr, splitstr)
    acc_list.append(accuracy_data)

    print(i, ": {:.2f}".format(acc_list[-1]))

print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test accuracy: {:.4f}, {}".format(
    np.mean(acc_list), np.round(np.std(acc_list), 2)))
