import sys, time, os
import numpy as np
import torch
import copy
import torch.nn as nn
import random
from copy import deepcopy
from tqdm import tqdm
from networks.Classifier import CLASSIFIER
import utils


class classifier_train(object):
  def __init__(self, CLASSI, args):
    super(classifier_train, self).__init__()
    self.args = args
    self.CLASSI = CLASSI
    #self.Classifier = Classifier

    #self.classifier = self.get_classifier()
    self.class_loss = torch.nn.CrossEntropyLoss().to(self.args.device)
    #self.class_optimizer = self.get_class_optimizer()
    #self.class_optimizer= torch.optim.Adam(self.CLASSI.parameters(), lr = 1e-4)
    self.class_optimizer= torch.optim.Adam(self.CLASSI.parameters(), weight_decay=self.args.e_class, lr = 1e-4)


  def train(self, task_id, traindata, trainlabels):
    train_data, train_label, train_attr, no_steps = utils.dataloader(traindata, trainlabels, traindata, self.args.test_batch_size)
    for e in range(self.args.class_epochs):
        loss = self.train_epoch(task_id, train_data, train_label, no_steps)
        print('epoch:', e + 1, 'class_loss:', loss)
    #self.save_class_model(task_id)

  def train_epoch(self, task_id, train_data, train_label, no_steps):
    self.CLASSI.train()
    #train_data, train_label, train_attr, no_steps = utils.dataloader(trainData, trainLabels, trainData, self.args.test_batch_size) 
    loss_sum = 0
    for i in range(no_steps):
      self.CLASSI.zero_grad()
      self.class_optimizer.zero_grad()
      batch_train, batch_label = train_data[i].to(self.args.device), train_label[i].to(self.args.device)

      out = self.CLASSI(batch_train)
      #print(torch.argmax(out, dim = 1), '7')
      loss = self.class_loss(out, batch_label)
      loss.backward(retain_graph = True)
      self.class_optimizer.step()
      loss_sum += loss
    return  loss_sum.item()

  def test(self, x):
    self.CLASSI.eval()
    with torch.no_grad():
      pred_s = torch.argmax(self.CLASSI(x), dim = 1)
      return pred_s
