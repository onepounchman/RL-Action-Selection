import numpy as np
import torch
import scipy
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Tuple
from torch.distributions import Normal

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def eval_vs(act_dim_true,index_set,extra):
     
    true_index = [i for i in range(act_dim_true)]
    p = act_dim_true
    n = extra
    tp = len([index for index in index_set if index in true_index])
    fp = len(index_set)-tp
    
    tpr = tp/p
    fpr = fp/(n+0.0000001)
    fdr = fp/(fp+tp+0.000001)
                                   
    return [tpr,fpr,fdr]


def create_mask(new_index,act_dim):

    # Create a tensor filled with zeros
    mask = torch.zeros(act_dim)

    # Set the elements at indices in the list to 1
    mask[new_index] = 1
    
    return mask


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


