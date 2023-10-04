import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser
import animal_hash
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from losses import pq_fun



class NoSampling(nn.Module):
  def __init__(self):
    super(NoSampling, self).__init__()

  def forward(self, pq):
    return torch.ones_like(pq)
  
    
class DRSampling(nn.Module):
  def __init__(self, params, M, Zq):
    super(DRSampling, self).__init__()
    self.gamma_drs = params["gamma_drs"]
    self.eps_drs = params["eps_drs"]
    self.M = M
    self.Zq = Zq

  def forward(self, pq=None):
    return torch.sigmoid(torch.log(pq/self.Zq) - torch.log(self.M) - torch.log(1-pq/self.M*np.exp(-self.eps_drs)) - self.gamma_drs)




class BudgetSampling(nn.Module):
  def __init__(self, params, M, Zq):
    super(BudgetSampling, self).__init__()
    self.budget = params['budget']
    self.c = 1
    self.M = M
    self.Zq = Zq


  # def compute_optimal_c(self, pq, n_iterations, depth=1, scale=1):
  #   if depth>30:
  #     return 1
  #   pqm = pq/self.M

  #   lr = 1e-1*scale
  #   with torch.no_grad():
  #     c = 1
  #     for n in range(n_iterations):
  #       grad = (pqm*(pqm*c<=1)).sum()*(torch.clamp(pqm*c, min=0, max=1) - self.budget).sum()
  #       c, c_prev = c - lr*(grad-1e-5), c
  #       if n % 3000 == 0 and n>0:
  #         lr /= 2 
  #       if abs(torch.mean(torch.clamp(pqm*c, min=0, max=1))-self.budget)<=5e-4:
  #         break
  #   logging.info(f'\tOptiC: n: {n}, \t c:{c:.2f}, gap:{abs(torch.mean(torch.clamp(pqm*c, min=0, max=1))-self.budget):.2f}, \t rate:{torch.mean(torch.clamp(pqm*c, min=0, max=1)):.2f}, grad: {grad:.2f}')
  #   logging.info(f'\tOptiC: scale: {scale:.3e}, depth: {depth}, lr:{lr:.3e}')


  #   if n == n_iterations-1:
  #     c_deep = self.compute_optimal_c(pq, n_iterations, depth=depth+1, scale=scale*0.5)
  #     if abs(torch.mean(torch.clamp(pqm*c, min=0, max=1))-self.budget)>abs(torch.mean(torch.clamp(pqm*c_deep, min=0, max=1))-self.budget):
  #       return c_deep
  #   return c

  def compute_optimal_c(self, pq, n_iterations, depth=1, scale=1):
    pqm = torch.Tensor((pq/self.M).cpu().numpy()).cuda()
    c_min = 1e-7
    c_max = 1e10
    c_med = (c_min+c_max)/2
    n_iteration = 0
    while n_iteration<n_iterations:
      if torch.mean(torch.clamp(pqm*c_med, min=0, max=1))-self.budget>1e-6:
        c_max = c_med
        c_med = (c_min+c_max)/2
      elif torch.mean(torch.clamp(pqm*c_med, min=0, max=1))-self.budget<-1e-6:
        c_min = c_med
        c_med = (c_min+c_max)/2
      else:
        n_iteration = n_iterations
      n_iteration += 1
      print(f'i: {n_iteration} c:{c_med:.2f}, bud: {torch.mean(torch.clamp(pqm*c_med, min=0, max=1)):.2f}')
    return torch.Tensor([c_med]).cuda()

  def update(self, pq):  
    n_update = 100 if self.training else 1000
  
    if self.budget is not None:
      c_opt = torch.clamp(self.compute_optimal_c(pq, n_update), min=1)
    else:
      c_opt = 1

    self.c = c_opt
    
  def forward(self, pq, update=False):
    pq = pq/self.Zq
    if update or self.training:
      self.update(pq)
    return torch.clamp(pq/self.M*self.c, min=0, max=1)


def get_sampling_function(config, sample, D):
  if config['sampling'] is None:
    return NoSampling().cuda(), 1
  else:
    rate = pq_fun(config)
    with torch.no_grad():
      M = torch.Tensor([0.]).cuda()
      pqs = torch.Tensor([0.]).cuda()
      while pqs.shape[0]<1000:
          x, y = sample()
          Dxf = D(x,y)
          pq = rate(Dxf)
          M = torch.max(torch.vstack((pq, M)))
          pqs = torch.vstack((pqs, pq))

      np.save("pqs", pqs.cpu().numpy())
      # Zq = torch.mean(pqs[1:])
      # M = M/Zq
      # print(Zq)
      M = torch.from_numpy(M .float().cpu().numpy()).cuda()

      if config['sampling'] == "DRS":
        return DRSampling(config, M, M/M).cuda(), M
      else:
        sampling = BudgetSampling(config, M, M/M).cuda()
        sampling.eval()
        sampling(pqs[1: 1000], update=True)
        return sampling, M
        


def get_sampler_function(config, sampling, D, sample):
  rate = pq_fun(config)
  n_samples = config['batch_size']
  if config['sampling'] is None:
    def RejectionSampler(test=False):
      with torch.no_grad():
        x_fake, y_fake = sample()
        if test:
          return  x_fake.shape[0], x_fake.shape[0]
        else:
          return x_fake, y_fake
    return RejectionSampler

  def RejectionSampler(test=False):
      with torch.no_grad():
        N = 0
        n_accepted = 0
        x_accepted = torch.Tensor().cuda()
        y_accepted = torch.Tensor().cuda()

        while n_accepted < n_samples:
                
                x_fake, y_fake = sample()
                Dxf = D(x_fake, y_fake)
                pq = rate(Dxf)
                a = sampling(pq)
                idx = a >= torch.rand((n_samples, 1)).cuda()
                n_accepted += int(idx.sum())
                x_accepted = torch.cat([x_accepted, x_fake[idx.view(-1)]], dim=0)
                y_accepted = torch.cat([y_accepted, y_fake[idx.view(-1)]], dim=0)

                N+=n_samples
        if test:
          return  N, n_accepted
        else:
          return x_accepted[:n_samples], y_accepted

  return RejectionSampler

def test_sampling(sampler):
  N = 0
  Na = 0
  while N<10000:
    n, na = sampler()
    N+=n
    Na+=na
  return N, Na


def plot_rejected(imgs, labels, D, name, sampling, dir, config):
    rate =  pq_fun(config)
    with torch.no_grad():
        torch.manual_seed(0)

        pq = rate(D(imgs, labels))
        a = sampling(pq)
        idx = a >= torch.rand((100, 1)).cuda()
    plt.figure(figsize=(12,12),  layout="compressed")
    for k in range(100):
        plt.subplot(10,10, k+1,)
        plt.imshow((imgs[k].cpu().permute(1,2,0).numpy()+1)/2)
        plt.gca().add_patch(plt.Rectangle((0,-0),imgs.shape[2]-1,imgs.shape[2]-1,
            edgecolor='green' if idx[k] else 'red',
            facecolor='none',
            lw=5))
        plt.xticks([])
        plt.yticks([])
    # plt.savefig(f'imgs/{name}.pdf')
    plt.savefig(f'{dir}/reject_{name}.png',dpi=100)

#     plt.tight_layout()
    plt.close()
