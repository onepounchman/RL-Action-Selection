"""
Main function for the sequential knockoffs (SEEK) algorithm

""" 

import sys
import numpy as np 
import pandas as pd
import random
import collections 
from sklearn.linear_model import Lasso,LassoCV
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    return standardized_data
    
class vs(object):
    
        def __init__(self,  q=0.1, k0=1, reg_method='randomforest', alpha=0.5, kf_plus=0, seed=2333,dataset='synthetic',extra=10,use_knockoff=False):
            """
            :param dataset: input data by format, pd.DataFrame(columns=['s1', 'a', 'r', 's2']).
            :param q: target FDR.
            :param reg_method: machine learning methods to compute the W-statistics, 'lasso', 'linear', or 'randomforest'.
            :param alpha: threshold for majority vote.
            :param kf_plus: If use knockoff+ (1) or not (0).
            :param seed: seed. 
            """
            self.q = 0.1
            self.k0 = k0
            self.reg_method = reg_method  
            self.alpha = alpha
            self.kf_plus = kf_plus
            self.seed = seed
            self.split = 5
            self.use_knockoff = use_knockoff
            self.extra = extra
            self.dataset = dataset

        
        def get_knockoff_list(self,replay_buffer,act_limit):
            
            
            self.buffer = replay_buffer
            seed = int(self.seed) 
            np.random.seed(seed)
            random.seed(seed)  

            num_splits = self.split

            G_set_all = []

            # variable selection based on knockoff

            sample_size = replay_buffer.ptr
            n_action = self.buffer.act_buf[0:sample_size,][0:2,].shape[1]
            n_state = self.buffer.obs_buf[0:sample_size,][0:2,].shape[1]
            
            if self.use_knockoff:
                W_all = np.empty(shape=(num_splits, n_action*2+n_state))
            else:
                W_all = np.empty(shape=(num_splits, n_action+n_state))
            
            for ith in range(num_splits):
                


                action_i = standardize(self.buffer.act_buf[0:sample_size,][ith::num_splits,])
                
                action_copy_i = standardize(self.buffer.act_copy_buf[0:sample_size,][ith::num_splits,])
                

                state_i = standardize(self.buffer.obs_buf[0:sample_size,][ith::num_splits,])

                next_state_i = standardize(self.buffer.obs2_buf[0:sample_size,][ith::num_splits,]) 

                reward_i = standardize(self.buffer.rew_buf[0:sample_size,][ith::num_splits,])
                

                n_state = state_i.shape[1]
                n_action = action_i.shape[1]
                
                
                Ghat = []


                X_train = action_i
                

                n=len(X_train) 

                
                if self.use_knockoff:
                    beta_mat = np.empty(shape=(n_state+1, n_state+n_action*2))
                    Xmat = np.concatenate((X_train,action_copy_i),1)
                    
                else:   
                    beta_mat = np.empty(shape=(n_state+1, n_state+n_action))
                    Xmat = X_train
                

                Xmat = np.concatenate((state_i,Xmat),axis=1)
                
                
                #compute the W-statistics
                if self.reg_method == 'lasso': 
                            
                    for ith_var in range(n_state+1):
                        if ith_var == 0:
                                yval = reward_i
                        else:
                            yval = next_state_i[:,ith_var-1]

                        alpha_list = np.exp(np.arange(-15,-8,1))  # Generates 50 values between 10^-3 and 10^1


                        lasso_cv = LassoCV(alphas=alpha_list, cv=5) 
                        lasso_cv.fit(Xmat, yval-np.mean(yval))
                        beta_mat[ith_var] = np.array(lasso_cv.coef_)

                    beta_mat = np.max(np.abs(beta_mat),0)
 

                elif self.reg_method == 'randomforest':  
                    for ith_var in range(n_state+1):
                        if ith_var == 0:
                            yval = reward_i
                        else:
                            yval = next_state_i[:,ith_var-1]

                        rf = RandomForestRegressor(n_estimators=int(n/10),max_features='sqrt') 
                        rf.fit(Xmat, yval)
                        

                        beta_mat[ith_var] = np.array((rf.feature_importances_).tolist()).reshape(1,-1)#[:,n_state:]
                    beta_mat = np.mean(np.abs(beta_mat),0)
                    
                W_all[ith,:] = beta_mat
                
            if not self.use_knockoff:
                    if self.reg_method == 'randomforest':
                        threshold = 0.1
                    elif self.reg_method == 'lasso':
                        threshold = 0.05

            
                    binary_array = (W_all[:,n_state:] > threshold).astype(int)
                
                    
                    selected = np.sum(binary_array, axis=0)
                    
                    indices = np.where(selected > num_splits*1/2)[0].tolist()
                    
                    return list(indices)
                
            else:
                    G_set_all= []
                    
                    for j in range(num_splits):
                        
                        max_beta = W_all[j,:]

                        Wi = (max_beta[n_state:n_state+n_action] - max_beta[n_state+n_action:])
    
                        if self.kf_plus == 0:
                            W_abs = np.sort(np.abs(Wi))
                            for i in range(len(W_abs)):
                                tt = W_abs[i]
                                if (np.sum(Wi <= -tt))/np.sum(Wi >= tt) <= self.q:
                                    break
                                
                            tau = tt

                        if self.kf_plus == 1:
                            W_abs = np.sort(np.abs(Wi))
                            for i in range(len(W_abs)):
                                    tt = W_abs[i]
                                    if (1 + np.sum(Wi <= -tt))/np.sum(Wi >= tt) <= self.q:
                                        break
                            tau = tt
                        if tau == 0:
                            Ghat = []
                        else:
                            Ghat = [i for i, x in enumerate((Wi >= tau).tolist()) if x == True]                
                        G_set_all.append(set(Ghat))


                    all_list = []
                    for iset in G_set_all:
                        all_list = all_list + list(iset) 
                    all_list
                    counter=collections.Counter(all_list)

                    vote_res = (np.array(list(counter.values())) / len(G_set_all) >= self.alpha).tolist() 
                    print(f'counter:{counter}')
                    print((np.array(list(counter.values())) / len(G_set_all)))
                    vote = []
                    for i in range(len(vote_res)):
                        if vote_res[i] == True:
                            vote.append(list(counter.keys())[i])

                    knockoff_list = vote  
                    
                    return knockoff_list
