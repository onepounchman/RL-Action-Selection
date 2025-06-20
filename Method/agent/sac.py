 
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import agent.core_sac as core
import os
import sys
import random
from utils import create_mask
import wandb
upstream_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(upstream_folder_path)


class sac(object):
    
    def __init__(self,observe_space,action_space,hidden_sizes,lr,gamma,alpha,device):
            """
            :param observe_space
            :param action_space
            :param lr: leanring rate
            :param gamma: 
            :param alpha:
            :param devide:
            """
            
            self.alpha = alpha
            self.gamma = gamma
            self.device = device
            
            self.polyak = 0.995
            
            self.ac = core.MLPActorCritic(observe_space, action_space,hidden_sizes).to(self.device)
            self.ac_targ = deepcopy(self.ac).to(self.device)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.ac_targ.parameters():
                p.requires_grad = False

            # List of parameters for both Q-networks (save this for convenience)
            self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

            # Set up optimizers for policy and q-function
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
            self.q_optimizer = Adam(self.q_params, lr=lr)
            
            self.mask = torch.ones(1,action_space.shape[0]).to(self.device)
            


            # Iterate through the named weights
        

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        q1 = self.ac.q1(o,a,self.mask)
        q2 = self.ac.q2(o,a,self.mask)
        
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2,self.mask)

            
            # feature selection
            
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2,self.mask)
            q2_pi_targ = self.ac_targ.q2(o2, a2,self.mask)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())
        
        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        
        o = data['obs']
        pi, logp_pi = self.ac.pi(o,self.mask)
        
        # feature selection
        
        q1_pi = self.ac.q1(o, pi,self.mask)
        q2_pi = self.ac.q2(o, pi,self.mask)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # log_pi shape batch_size

        # Entropy-regularized policy loss
        # logp_pi and q_pi shape [256] batch size

        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
        
        entropy = (-logp_pi).mean().item()
        return loss_pi, entropy

    # Set up model saving
    #logger.setup_pytorch_saver(ac)

    def update(self,data):

        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()

        self.q_optimizer.step()
        
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, entropy = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        
        log_dict = {'loss_pi':loss_pi.item(),'loss_q':loss_q.item(),'entropy': entropy}
        wandb.log(log_dict)

        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        
        
                
    def get_action(self,o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), self.mask,
                      deterministic)
    
    def update_mask(self,new_mask):
        
        self.mask = new_mask.to(self.device)
        

    

        

        



