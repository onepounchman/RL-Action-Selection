from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import agent.core_ppo as core
import os
import sys
import random
from utils import create_mask
import wandb

upstream_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(upstream_folder_path)


class ppo(object):
    
    def __init__(self,observe_space,action_space,hidden_sizes,pi_lr,vf_lr,iters,clip_ratio,gamma,device):
            
            self.device = device
            self.clip_ratio = clip_ratio
            self.gamma=gamma
            self.target_kl=0.01
            self.ac = core.MLPActorCritic(observe_space, action_space,hidden_sizes).to(self.device)
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
            self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
            self.train_iters = iters
            self.mask = torch.ones(1,action_space.shape[0]).to(self.device)
            self.device = device

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self,data):
        
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act, self.mask)
        
    
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self,data):
        obs, ret = data['obs'], data['ret']
        
        return ((self.ac.v(obs) - ret)**2).mean()

    
    def update(self,buf):
        
        data = buf.get(self.device)
        
        pi_l_old, pi_info_old= self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        loss_pi_all = []
        for i in range(self.train_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            loss_pi_all.append(loss_pi.item())
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        
        loss_pi_mean = sum(loss_pi_all)/len(loss_pi_all)

        
        # Value function learning
        loss_v_all = []
        for i in range(self.train_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v_all.append(loss_v.item())
            loss_v.backward()
            self.vf_optimizer.step()
        loss_v_mean = sum(loss_v_all)/len(loss_v_all)
        
        log_dict = {'loss_pi':loss_pi_mean,'loss_v':loss_v_mean}

        
    def get_action(self,o):
        return self.ac.step(torch.as_tensor(o, dtype=torch.float32).to(self.device), self.mask)
    
    def update_mask(self,new_mask):
        
        self.mask = new_mask.to(self.device)




