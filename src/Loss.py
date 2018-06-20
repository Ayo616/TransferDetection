import torch
import torch.nn as nn
import numpy as np

def TONLoss(input, output):
    return None

def CORAL(source, target):
    d = source.data.shape[1]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss

def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    #loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    #delta = f_of_X - f_of_Y
    #loss = torch.mean((delta * delta).sum(1))
    #print(loss)
    return loss


def Euclidean(point,matrix):
    from scipy.spatial.distance import pdist,squareform
    X=np.vstack([point,matrix])
    d2=pdist(X,'euclidean')
    z = squareform(d2)
    return (np.mean(z,axis=1)[0:4])


class CombinedLoss(nn.Module):

    def __init__(self):
        super(CombinedLoss,self).__init__()
        return

    def forward(self,*my_args):
        return sum(my_args)
