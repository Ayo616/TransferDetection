import torch
import torch.nn as nn
from torch.autograd import Function, Variable

'''
# Define the network

Transefer outlier learning network

'''

class TOLN(nn.Module):
    def __init__(self):
        super(TOLN, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16,30),
            nn.Tanh(),
            nn.Linear(30,60)
        )

        self.decode = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(60,30),
            nn.Tanh(),
            nn.Linear(30,16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8,1)
        )

    def forward(self,source,target):

        source = self.encode(source)
        target = self.encode(target)

        source = self.decode(source)
        target = self.decode(target)

        return  source , target





