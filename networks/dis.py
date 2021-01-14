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

class Discriminator(nn.Module):
  def __init__(self, args, task_id):
    super(Discriminator, self).__init__()
    self.args = args
    self.task_id = task_id
    if args.diff == 'yes':
      self.dis = torch.nn.Sequential(
          GradientReversalLayer(args.lam),
          torch.nn.Linear(args.n_x + args.n_y, args.hid),
          torch.nn.LeakyReLU(),
          torch.nn.Linear(args.hid, task_id + 2)
      )
    else:
      self.dis = torch.nn.Sequential(
          torch.nn.Linear(args.n_x + args.n_y, args.hid),
          torch.nn.LeakyReLU(),
          torch.nn.Linear(args.hid, task_id + 2)
      )
  def forward(self, x):
    return self.dis(x)

class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversalLayer(nn.Module):
  def __init__(self, lambda_):
    super(GradientReversalLayer, self).__init__()
    self.lambda_ = lambda_

  def forward(self, x):
    return GradientReversalFunction.apply(x, self.lambda_)
