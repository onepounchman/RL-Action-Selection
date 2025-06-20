from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
from time import time
import os
import sys
from agent import ppo
import random
from Method.utils import combined_shape,create_mask,count_vars,discount_cumsum,eval_vs
from vs import vs
import gc
import wandb
import inspect
import json



agent_list = {'ppo': ppo,'ppo_kf':ppo}


class vsBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.act_copy_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, obs_next, act, act_copy, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs_next
        self.act_buf[self.ptr] = act
        self.act_copy_buf[self.ptr] = act_copy
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def get(self,device = 'cpu'):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}
    
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.act_copy_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, obs_next, act, act_copy, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs_next
        self.act_buf[self.ptr] = act
        self.act_copy_buf[self.ptr] = act_copy
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self,device = 'cpu'):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}


def action_sample(o,agent,env,extra,act_limit,prev_mask):
    a, v, logp = agent.get_action(o)
    a = a.reshape((-1,))
    return a,v,logp


def train_ppo(env_fn, agent_type = 'ppo', hidden_sizes = [64,32], seed=0, local_steps_per_epoch = 4000,epochs=250,num_test_episodes = 10, 
        gamma=0.99, pi_lr=1e-3,vs_lr=3e-4,lam=0.97, clip_ratio=0.1,iters=10,max_ep_len=1000,logger_kwargs=dict(),env_name='None',
        extra=0,use_vs=None,use_knockoff = True, vs_epoch = 4, use_wandb=True,gpu_id='0',use_cuda=False,adaptive=False):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    time_0 = time()

    # Random seed
    if use_cuda:
            #os.environ["CUDA_VISIBLE_DEVICES"]= f"{gpu_id}"
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
    agent_class = agent_list[agent_type]


    agent = agent_class(env.observation_space,action_space,hidden_sizes,pi_lr,vs_lr,iters,clip_ratio,gamma,device)
    

    train_result =[]
    test_result =[]
    select_steps = max_ep_len*vs_epoch
    ep_ret, ep_len =0, 0  


    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    buf_vs = vsBuffer(obs_dim, act_dim, select_steps, gamma, lam)

    o, info = env.reset(seed=seed)

    ep_ret, ep_len = 0, 0
    
    train_result =[]
    test_result =[]
    
    prev_mask = create_mask([k for k in range(act_dim)],act_dim)
    vs_flag = use_vs

    
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            if agent_type =='ppo_lattice' or agent_type =='ppo_gsde':
                agent.ac.pi.lattice.sample_weights(agent.ac.pi.log_std)
            a, v, logp = action_sample(o,agent,env,extra,act_limit,prev_mask)
            a_copy, _, _ = action_sample(o,agent,env,extra,act_limit,prev_mask)

            next_o, r, d, _ ,_= env.step(a)

            ep_ret += r
            ep_len += 1

            buf.store(o, next_o,a, a_copy, r, v, logp)
            
            if buf_vs.ptr<select_steps:
                 buf_vs.store(o, next_o,a, a_copy, r, v, logp)
            
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ =   agent.get_action(o)
                else:
                    v = 0
                buf.finish_path(v)
                o, info = env.reset()
                ep_ret, ep_len =  0, 0

        
        if buf_vs.ptr==select_steps and vs_flag:
     

                vs_model = vs(q=0.1, k0=1, reg_method='lasso', alpha=0.5, kf_plus=0,seed=2333,dataset=dataset,extra=extra,use_knockoff=use_knockoff)
                
                index_set = vs_model.get_knockoff_list(buf_vs,act_dim)  
                index_set.sort()
                print(index_set)
                os.makedirs(os.getcwd()+f'/{env_name}_experiments/results_selection', exist_ok=True)
                file_path_ks = os.getcwd()+f'/{env_name}_experiments/results_selection/{env_name}_{buf_vs.ptr}_{seed}.json'
                with open(file_path_ks, 'w') as f:
                    json.dump(index_set, f)
                    

                eval_results = eval_vs(act_dim_true,index_set,extra)
      
                if use_wandb:
                    log_dict = {'tpr': eval_results[0],
                                'fpr': eval_results[1],
                                'fdr': eval_results[2],
                                'select_act_prop': len(index_set)/act_dim_true,
                                }
                    wandb.log(log_dict)

                mask = create_mask(index_set,act_dim)
            
                prev_mask = mask

                vs_flag = False
        
        agent.update(buf)
        
        agent.update_mask(prev_mask)

        reward_test,len_test = test_agent(agent,test_env,max_ep_len,act_dim_true,prev_mask,epoch*seed)
        test_result.append(reward_test)
        

        if use_wandb:
                log_dict = {
                            'test_reward': reward_test,
                            'test_epi_length':len_test
                        }

                wandb.log(log_dict)
    

    del buf,agent
    torch.cuda.empty_cache()
    gc.collect()
    return train_result,test_result
                    
        

def test_agent(agent,test_env,max_ep_len,act_dim_true,mask,seed,num_test_episodes=5):
        reward_all = 0
        ep_len_all = 0
        for j in range(num_test_episodes):
            o, info = test_env.reset(seed=seed) 
            d, ep_ret_test, ep_len_test  = False, 0, 0
            while not(d or (ep_len_test == max_ep_len)): 
                a, _, _ =  agent.get_action(o)
                real_a = a[0:act_dim_true].reshape((act_dim_true,))
                o, r, d, _,_ = test_env.step(real_a)
                ep_ret_test += r
                ep_len_test += 1
                
            reward_all += ep_ret_test
            ep_len_all += ep_len_test
        return reward_all/num_test_episodes,ep_len_all/num_test_episodes
    
