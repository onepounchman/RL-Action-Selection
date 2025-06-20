from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
import sys
import random
from agent import sac
from Method.utils import combined_shape,create_mask,eval_vs
from vs import vs
import gc

upstream_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(upstream_folder_path)


import wandb


agent_list = {'sac': sac,'sac_kf': sac }

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.act_copy_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, act_copy, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.act_copy_buf[self.ptr] = act_copy
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32,device='cpu'):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}


def action_sample_after(o,agent,env,act_limit,prev_mask):
    a = agent.get_action(o).ravel()
    
    return a 

def action_sample_initial(env,act_limit,prev_mask):
    a = env.action_space.sample()
    return a 


def train_sac(env_fn, agent_type = 'sac',hidden_sizes = [256,256], seed=0, 
         epochs=100, num_test_episodes =20,replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=200, update_every=50, max_ep_len=200,
        logger_kwargs=dict(), save_freq=1,env_name='None',extra=0,mu=0,use_vs=None,vs_epoch=4,use_knockoff=True,use_wandb=True,gpu_id='0',use_cuda=False):
    

    if use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f'gpu id {gpu_id}')

    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    
    print(f'device {device}')
    dataset = env_name
    extra = extra
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  
    torch.cuda.manual_seed_all(seed)

    env, test_env = env_fn(), env_fn()
    env.action_space.seed(seed)
    test_env.action_space.seed(seed+100)
    
    obs_dim = env.observation_space.shape
    act_dim_true = env.action_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    
    action_space = env.action_space 


    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    
    agent_class = agent_list[agent_type]

    agent = agent_class(env.observation_space,action_space,hidden_sizes,lr,gamma,alpha,device)
        

    train_result =[]
    test_result =[]
    total_steps = epochs*max_ep_len
    select_steps = max_ep_len*vs_epoch
    ep_ret, ep_len =0, 0  
    o, info = env.reset(seed=seed)
    
    vs_flag = use_vs

        
    index_set = None
    prev_mask = create_mask([k for k in range(act_dim)],act_dim)
    
    for t in range(total_steps):

            if  replay_buffer.ptr > start_steps:
                a = action_sample_after(o,agent,env,act_limit,prev_mask)
                a_copy = action_sample_after(o,agent,env,act_limit,prev_mask)
                
            else:
                a = action_sample_initial(env,act_limit,prev_mask)
                a_copy = action_sample_initial(env,act_limit,prev_mask)
                

            o2, r, d, _,_ = env.step(a)
            
            ep_ret += r
            ep_len += 1

            d = False if ep_len==max_ep_len else d


            replay_buffer.store(o, a, a_copy, r, o2, d)

            o = o2


            if d or (ep_len == max_ep_len):
                train_result.append(ep_ret)
                o, info = env.reset()
                ep_ret, ep_len = 0, 0

                
            if (t+1) % max_ep_len == 0:

                reward_test,len_test = test_agent(agent,test_env,max_ep_len,act_dim_true,(t+1)% max_ep_len*seed)
                test_result.append(reward_test)
                
                if use_wandb:
                    log_dict = {
                            'test_reward': reward_test,
                            'test_epi_length':len_test
                        }

                    wandb.log(log_dict)


                if replay_buffer.ptr==select_steps and vs_flag:
                    
                    vs_model = vs(q=0.1, k0=1, reg_method='lasso', alpha=0.5, kf_plus=0,seed=2333,dataset=dataset,extra=extra,use_knockoff=use_knockoff)

                    index_set = vs_model.get_knockoff_list(replay_buffer,act_dim)  

                    index_set.sort()

                    mask = create_mask(index_set,act_dim)
                    
                    prev_mask = mask

                    print(f'index set :{index_set}')
                    
                    eval_results = eval_vs(act_dim_true,index_set,extra)
                    
                    
                    if use_wandb:
                        log_dict = {'tpr': eval_results[0],
                                    'fpr': eval_results[1],
                                    'fdr': eval_results[2],
                                    'select_act_prop': len(index_set)/act_dim_true,
                                    }
                        wandb.log(log_dict)

                    vs_flag = False
                    agent.update_mask(mask)
                    
                    
            if replay_buffer.size > start_steps and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size,device)
                    agent.update(data=batch)

    del replay_buffer,agent
    torch.cuda.empty_cache()
    gc.collect()
    return train_result,test_result


def test_agent(agent,test_env,max_ep_len,act_dim_true,seed,num_test_episodes=10):
        reward_all = 0
        ep_len_all = 0
        for j in range(num_test_episodes):
            o, info = test_env.reset(seed=seed) 
            d, ep_ret_test, ep_len_test  = False, 0, 0
            while not(d or (ep_len_test == max_ep_len)):
                a = agent.get_action(o).reshape(-1,1)
                real_a = a[0:act_dim_true].reshape((act_dim_true,))
                o, r, d, _,_ = test_env.step(real_a)
                ep_ret_test += r
                ep_len_test += 1
                
            reward_all += ep_ret_test
            ep_len_all += ep_len_test
        return reward_all/num_test_episodes,ep_len_all/num_test_episodes
    

    