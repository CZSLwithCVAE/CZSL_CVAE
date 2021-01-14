import os,argparse,time
import numpy as np
#from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
import random

def dataloader(x, y, attr, batch_size):
  #x = x.detach().numpy()
  #length = x.shape[0]
  length = x.size()[0]
  indices = np.arange(length)
  random.shuffle(indices)
  new_x = x[indices]
  new_y = y[indices]
  new_attr = attr[indices]
  #print(indices)
  #print(new_y)
  x_list = [[] for ii in range(int(length / batch_size) + 1)]
  y_list = [[] for ii in range(int(length / batch_size) + 1)]
  attr_list = [[] for ii in range(int(length / batch_size) + 1)]

  for j in range(len(x_list)):
    for i in range(j * batch_size, (j + 1) * batch_size):
      if i >= length:
        break
      else:
        x_list[j].append(list(new_x[i].detach().numpy()))
        #print(new_y[i], 'new_y', i)
        y_list[j].append(new_y[i])
        attr_list[j].append(list(new_attr[i].detach().numpy()))

    x_list[j] = torch.tensor(x_list[j])
    y_list[j] = torch.tensor(y_list[j])
    attr_list[j] = torch.tensor(attr_list[j])
  #print(x_list[1], '111', y_list[1], attr_list[1])
  return (x_list, y_list, attr_list, len(x_list))
#print(x.shape, dataloader(x, args))

def get_model(model):
  return deepcopy(model.state_dict())
