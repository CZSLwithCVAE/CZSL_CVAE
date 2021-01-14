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

class encoder(nn.Module):
  def __init__(self, args):
    super(encoder, self).__init__()
    self.args = args
    self.lin1 = nn.Linear(args.n_x + args.n_y, args.hid)
    self.lin2 = nn.Linear(args.hid, 2 * args.n_z)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.lin1(x)
    x = self.relu(x)
    x = self.lin2(x)
    x = self.relu(x)
    #print(x, 'x')
    mu = x[:, 0:self.args.n_z]
    sigma = x[:, self.args.n_z:]
    return mu, sigma

class decoder(nn.Module):
  def __init__(self, args):
    super(decoder, self).__init__()
    self.lin1 = nn.Linear(args.n_z + args.n_y, args.hid)
    self.lin2 = nn.Linear(args.hid, args.n_x)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.lin1(x)
    x = self.relu(x)
    x = torch.sigmoid(self.lin2(x))
    return x


class shared(nn.Module):
  def __init__(self, args):
    super(shared, self).__init__()
    self.encoder = encoder(args)
    self.decoder = decoder(args)

  def forward(self, x, attr):
    mu, sigma = self.encoder(x)
    #print(mu, 'mu', sigma)
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(std)
    z = mu + std * eps
    z = torch.cat([z, attr], dim = 1)
    #print(z, 'z')
    x_pred = self.decoder(z)
    #print(x_pred, 'x_pred')
    return x_pred, mu, sigma

class vae(nn.Module):
  def __init__(self, args):
    super(vae, self).__init__()
    self.encoder = encoder(args)
    self.decoder = decoder(args)

  def forward(self, x, attr):
    mu, sigma = self.encoder(x)
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(std)
    z = mu + std * eps
    z = torch.cat([z, attr], dim = 1)
    x_pred = self.decoder(z)
    return x_pred, mu, sigma

class private(nn.Module):
  def __init__(self, args):
    super(private, self).__init__()
    self.task = torch.nn.ModuleList()
    for _ in range(args.num_tasks):
      self.task.append(vae(args))
    
  def forward(self, x, attr, task_id):
    return self.task[task_id].forward(x, attr)
    #return torch.stack([self.task[task_id].forward(x[i], attr) for i in range(x.size(0))])

class net(nn.Module):
  def __init__(self, args):
    super(net, self).__init__()
    self.shared = shared(args)
    self.private = private(args)
    self.head = torch.nn.ModuleList()
    for _ in range(args.num_tasks):
      self.head.append(
          torch.nn.Sequential(
              torch.nn.Linear(2 * args.n_x, args.hid),
              torch.nn.ReLU(),
              torch.nn.Dropout(p = 0.2),
              torch.nn.Linear(args.hid, args.num_classes),
              #torch.nn.Softmax(dim = 1)
          )
      )
  def forward(self, x, attr, task_id):
    x_s, mu_s, sigma_s = self.shared(x, attr)
    x_p, mu_p, sigma_p = self.private(x, attr, task_id)
    #print(x_s,'x_s', x_p, 'c')
    x = torch.cat([x_s, x_p], dim = 1)
    return self.head[task_id].forward(x)
    #return torch.stack([self.head[task_id].forward(x[i]) for i in range(x.size(0))])

  def out_of_shared(self, x, attr):
    return self.shared(x, attr)

  def out_of_private(self, x, attr, task_id):
    return self.private(x, attr, task_id)


  def decoder_of_shared_and_private(self, z, attr, task_id):
    z_attr = torch.cat([z, attr], dim = 1)
    return self.shared.decoder(z_attr), self.private.task[task_id].decoder(z_attr)