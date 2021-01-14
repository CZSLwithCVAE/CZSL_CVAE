import sys, time, os
import numpy as np
import torch
import copy

from copy import deepcopy
from tqdm import tqdm
from networks.dis import Discriminator
import utils

class A_VAE(object):
  def __init__(self, model, args):
    super(A_VAE, self).__init__()
    self.args = args
    self.model = model
    #self.network = network
    self.discriminator = self.get_discriminator(0)

    self.task_loss = torch.nn.CrossEntropyLoss().to(self.args.device)
    self.adv_loss_model = torch.nn.CrossEntropyLoss().to(self.args.device)
    self.adv_loss_d = torch.nn.CrossEntropyLoss().to(self.args.device)

    self.diff_loss = DiffLoss().to(self.args.device)
    self.vae_loss = vaeloss()

    self.model_optimizer = self.get_model_optimizer(0)
    self.d_optimizer = self.get_dis_optimizer()

    self.mu = 0.
    self.sigma = 1.

  def get_discriminator(self, task_id):
    discriminator = Discriminator(self.args, task_id).to(self.args.device)
    return discriminator

  def get_model_optimizer(self, task_id):
    model_optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
    return model_optimizer

  def get_dis_optimizer(self):
    d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = 1e-3)
    return d_optimizer

  def train(self, task_id, trainData, trainLabels, trainLabelsVectors):
    self.discriminator = self.get_discriminator(task_id)
    self.model_optimizer = self.get_model_optimizer(task_id)
    self.d_optimizer = self.get_dis_optimizer()

    train_data, train_label, train_attr, no_steps = utils.dataloader(trainData, trainLabels, trainLabelsVectors, self.args.train_batch_size)


    for e in range(self.args.num_epochs):
      loss = self.train_epoch(task_id, train_data, train_label, train_attr, no_steps)
      print('epoch', e + 1, 'Total loss:', loss)



    #self.save_all_model(task_id)

  def train_epoch(self, task_id, train_data, train_label, train_attr, no_steps):
    self.model.train()
    self.discriminator.train()
    loss_sum, loss_task_sum, loss_p_sum, loss_s_sum, adv_loss_sum, d_fake_s, d_real_s = 0, 0, 0, 0, 0, 0, 0

    #train_data, train_label, train_attr, no_steps = utils.dataloader(trainData, trainLabels, 
     #                                                          trainLabelsVectors, self.args.train_batch_size)
    
    for i in range(no_steps):
      batch_train, batch_label, batch_attr = train_data[i].to(self.args.device), train_label[i].to(self.args.device), train_attr[i].to(self.args.device)
      x_attr = torch.cat([batch_train, batch_attr], dim = 1)

      for _ in range(self.args.s_steps):
        self.model_optimizer.zero_grad()
        self.model.zero_grad()
        out_model = self.model(x_attr, batch_attr, task_id)

        x_s, mu_s, sigma_s = self.model.shared(x_attr, batch_attr)
        x_p, mu_p, sigma_p = self.model.private(x_attr, batch_attr, task_id)

        task_loss = self.task_loss(out_model, batch_label)
        #------- shared loss ------------#
        recon_loss_s = torch.mean((x_s - batch_train)**2)
        loss_s = self.vae_loss.forward(recon_loss_s, mu_s, sigma_s)
        # ------------ vae loss from private module --------------------#
        recon_loss_p = torch.mean((x_p - batch_train)**2)
        loss_p = self.vae_loss.forward(recon_loss_p, mu_p, sigma_p)
        #------------- adv loss from discriminator ---------------------#
        d_real = (task_id + 1) * torch.ones(len(batch_train), dtype = torch.int64).to(self.args.device)
        d_fake = torch.zeros_like(d_real).to(self.args.device)
        x_s_attr = torch.cat([x_s, batch_attr], dim = 1)
        #print(x_s_attr.shape, '1') 
        dis_out = self.discriminator(x_s_attr)
        adv_loss = self.adv_loss_model(dis_out, d_real)
        if self.args.diff == 'yes':
          diff_loss = self.diff_loss(x_s, x_p)
        else:
          diff_loss = 0

        loss = task_loss + loss_p + loss_s + self.args.adv * adv_loss #+ self.args.orth * diff_loss

        loss.backward(retain_graph = True)
        self.model_optimizer.step()
      
      for _ in range(self.args.d_steps):
        self.d_optimizer.zero_grad()
        self.discriminator.zero_grad()

        out_model_d = self.model(x_attr, batch_attr, task_id)
        x_s_d, _, _ = self.model.shared(x_attr, batch_attr)
        dis_in = torch.cat([x_s_d, batch_attr], dim = 1)
        #print(x_s_d.shape, '2')
        dis_real = self.discriminator(dis_in)
        real_loss = self.adv_loss_d(dis_real, d_real)
        real_loss.backward(retain_graph = True)

        z_fake=torch.as_tensor(np.random.normal(0., 1., (len(batch_train),self.args.n_x)),dtype=torch.float32).to(self.args.device)
        z_attr = torch.cat([z_fake, batch_attr], dim = 1)
        #print(z_attr.shape, '2')
        d_out_fake = self.discriminator(z_attr)
        adv_loss_fake = self.adv_loss_d(d_out_fake, d_fake)
        adv_loss_fake.backward(retain_graph = True)
        self.d_optimizer.step()
      
      loss_sum += loss
      loss_p_sum += loss_p
      loss_s_sum += loss_s
      loss_task_sum += task_loss
      adv_loss_sum += adv_loss
      d_fake_s += adv_loss_fake
      d_real_s += real_loss

    return (loss.item(), loss_p_sum.item(), loss_s_sum.item(), loss_task_sum.item(), adv_loss_sum.item(), d_fake_s.item(), d_real_s.item()) 
    
  def test(self, z, attr, task_id):
    with torch.no_grad():
      #for param in self.model.parameters():
        #print(param[0], 'para')
      share_out, private_out = self.model.decoder_of_shared_and_private(z, attr, task_id)
      return share_out


class vaeloss():
  def __init__(self):
    super(vaeloss, self).__init__()

  def forward(self, recon_loss, mu, sigma):
    kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    #print('kl_div', kl_div.item())
    return recon_loss + kl_div

class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))