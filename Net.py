# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 01:08:24 2018

@author: Sandile
"""
import torch
import torch.nn as nn
import torch.nn.functional as  F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 7)
        self.fc2 = nn.Linear(7, 4)

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(x)
        return x


nn = Net()
print(nn)
