#!/usr/bin/env python
import myosuite
from functools import partial
from time import time
import yaml
from agent.sac import sac 
import argparse
from multiprocessing import Process
import json 
import sys
import os
from Method.train_sac import train_sac
from Method.train_ppo import train_ppo
from Env import AntEnv_high,HalfCheetahEnv_high,HopperEnv_high
import wandb
upstream_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(upstream_folder_path)

method_class_list = {'sac':train_sac,'sac_kf':train_sac,'ppo':train_ppo,'ppo_kf':train_ppo}
env_list = {'Ant-high': AntEnv_high,'HalfCheetah-high':HalfCheetahEnv_high,'Hopper':HopperEnv_high}

print(gym.__version__)

def main(env_name,method,gpu_id,args):
                ''' Test training a small agent in a simple environment '''
                print(env_name)
                keys = ['epochs','vs_epoch','alpha','use_vs','use_knockoff','extra']
                with open(f'config/{method}/{env_name}.yaml', 'r') as yaml_file:
                    config = yaml.safe_load(yaml_file)
                model_para = config['model']
                model_para['epochs'] = args.epochs
                seeds = [i for i in range(5)]
   

                use_vs = args.use_vs
                use_kf = args.use_kf
                use_wandb = args.use_wandb
                use_cuda = args.use_cuda
                vs_epoch = args.vs_epoch
                extra = args.extra

                env_fn = partial(env_list[env_name],extra)

                if use_wandb:
                    wandb.login(key='your key')
                
                model_para['agent_type'] = f'{method}'
                model_para['use_wandb'] = use_wandb
                model_para['gpu_id'] = gpu_id
                model_para['env_name'] = env_name
                model_para['env_fn'] = env_fn 
                model_para['use_cuda'] = use_cuda
                model_para['extra'] = extra

                if not use_vs and use_kf:
                        print('not use vs but use kf')

                model_para['use_vs'] = use_vs
    
                model_para['vs_epoch'] = vs_epoch
                model_para['use_knockoff'] = use_kf
                para_config = model_para
                para_formatted_string = f"{method}_{env_name}_"+"_".join([f"{key}={value}" for key, value in model_para.items() if key in keys])
                train_result_total = []
                test_result_total = []

                for seed in seeds:
                    model_para['seed'] = seed
                    if use_wandb:
                        name = f'{method}-{env_name}-vs={use_vs}-use_kf={use_kf}-{seed}'
                        wandb.init(project='your project name', name=name,config=para_config)

                    model = method_class_list[f'{method}']
                    train_result,test_result= model(**model_para)
                    train_result_total.append(train_result)
                    test_result_total.append(test_result)
        
                    if use_wandb:
                        wandb.finish()

                folder_path= os.getcwd()+f'/{env_name}_experiments/results'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path_test = f'{folder_path}/'+para_formatted_string+f'_test.json'

                with open(file_path_test, 'w') as json_file:
                     json.dump(test_result_total, json_file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--extra", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vs_epoch", type=int, default=4)
    parser.add_argument("--method", type=str, default='sac')
    parser.add_argument("--use_vs",  action="store_true")
    parser.add_argument("--use_kf",  action="store_true")
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--use_cuda",  action="store_true")
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    env_name = args.env
    gpu_id = args.gpu_id
    method = args.method
    if method=='ppo_kf' or method=='sac_kf':
        args.use_kf=True
        args.use_vs=True
    
    
    main(env_name,method,gpu_id,args)
    
