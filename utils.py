
import time
import random
import math
import numba as nb
import numpy as np

@nb.njit
def repeat(stim_pattern, stim_dur):
    '''repeat function similar to np.repeat written to be used with numba'''
    stim_len = len(stim_pattern)
    total_dur = stim_len * stim_dur
    stim = np.zeros((total_dur))
    idx = 0
    for i in range(stim_len):
        stim[idx:idx+stim_dur] = stim_pattern[i]
        idx +=stim_dur
    return stim

@nb.njit
def create_synapse(value,post,pre,Nconn):
    '''Function to create synapse with given num of pre and post connections with synaptic strength value
       and num: of connections, Nconn'''
    w = np.zeros((post,pre))
    C = np.zeros((post,pre))
    for i in range(post):
        Connection = np.random.choice(np.arange(0,pre), Nconn, replace=False)
        for j in Connection:
            C[i,j] = 1
            w[i,j] = value/Nconn
    return w, C

@nb.njit
def test_stimulus(stim_strength,dt):
    '''Function to generate and return test stimulus for visual and motor'''
    stim_V = repeat(np.array((0,stim_strength,0,0,0,stim_strength,0)), int(1000/dt))
    stim_M = repeat(np.array((0,stim_strength,0,stim_strength,0,0,0)), int(1000/dt))
    return stim_V, stim_M

@nb.njit
def train_stimulus(stim_value,stim_count,dt, zero_v):
    '''Function to generate and return train stimulus for visual and motor'''
    stim_times = np.ones((stim_count))
    stim_times[::2] =0
    stimulus = stim_times * np.random.uniform(0.0,stim_value,size=stim_count)
    time = 1
    m = repeat(stimulus,10000)
    non_zero_m = np.nonzero(stim_times)[0]
    stim_times[np.random.choice(non_zero_m, zero_v)] = 0
    v = repeat(stim_times * stimulus,10000)
    return m,v

@nb.njit
def arr_conv(a_list):
    '''Function to convert list of arrays to array with numba'''
    r = len(a_list)
    c = len(a_list[0])
    new_arr = np.zeros((r,c))
    for i in range(r):
        new_arr[i] = a_list[i]
    return(new_arr)