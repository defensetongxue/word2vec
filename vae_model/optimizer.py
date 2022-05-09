import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes):
    cost = F.mse_loss(preds,labels)
    # print(cost)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    # print("cost ",cost,' ',"KDL ",KLD)
    return cost + KLD
