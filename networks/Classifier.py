import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from copy import deepcopy
from sklearn.preprocessing import normalize
import glob, os

class CLASSIFIER(nn.Module):
  def __init__(self, args):
    #print('args', args)
    super(CLASSIFIER, self).__init__()
    self.lin1 = nn.Linear(args.n_x, args.hid)
    self.rel = nn.ReLU()
    self.drop = nn.Dropout(p = 0.2)
    self.lin2 = nn.Linear(args.hid, args.num_classes)
    #self.soft = nn.Softmax(dim = 1)
  def forward(self, x):
    x = self.rel(self.lin1(x))
    x = self.drop(x)
    x = self.lin2(x)
    #x = self.soft(x)
    return x