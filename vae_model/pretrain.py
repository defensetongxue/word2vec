
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from progress.bar import Bar

def train_step(model, optimizer, target, features, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, 1)
    loss_train = F.mse_loss(output[idx_train], target[idx_train])
    # print(loss_train)
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


def validate_step(model, target, features, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features, 1)
        loss_val = F.mse_loss(output[idx_val], target[idx_val])
        return loss_val.item()


class do_pretrain_job():
    def __init__(self, tokenizer, lr=0.02, epochs=2000,patience=100,nheads=1) -> None:
        self.tokenizer = tokenizer
        self.lr = lr
        self.nheads=nheads
        self.epochs = epochs
        self.patience=patience
    def __call__(self, model,  features, idx_train):
        print("begin to do pretrain jobs with epoches {}".format(self.epochs))
        
        target = self.tokenizer(features).repeat(1,self.nheads)
        target = torch.tensor(target)
        optimizer_pretrain = optim.Adam(model.parameters(),
                               lr=self.lr)
        best = 99999
        bad_cnt=0
        
        bar = Bar('pretrain', max=self.epochs)
        check_file="pretrained/prebest.pt"
        for epoch in range(self.epochs):
            
            loss_tra = train_step(
                model, optimizer_pretrain, target, features, idx_train)
            if loss_tra < best:
                best = loss_tra
                bad_cnt=0
                torch.save(model.state_dict(), check_file)
            else:
                bad_cnt+=1
            if bad_cnt==self.patience:
                break
            # if (epoch+1) % 100 == 0:
            #     print('Epoch:{:04d}'.format(epoch+1),
            #           'loss:{:.5f}'.format(loss_tra))
            bar.next()
        bar.finish()
        print("finished training")
        if bad_cnt==self.patience:
            print("收敛！！！")
        else:
            print("还没有收敛")
        return 
