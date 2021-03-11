import time
import random
import math
import numba as nb
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from utils import *


@nb.njit
def Simulate(weight_dict, neurons_dict, rates_dict, stim_dict, eta_dict, flag_dict, stim_params, fixed_inp, rates,
             weights):
    '''Simulate the positive prediction error neurons.

    Arguments:
        weight_dict : dictionary with all the weights of network
        neurons_dict: dictionary with number of exc and inh neurons
        rates_dict  : dictionary with baseline firing rates of all neurons
        stim_dict   : dictionary specifying the type of input stimulus to each neuron
        eta_dict    : dictionary specifying the learning rate params in plastic network
        flag_dict   : dictionary with different flags - test for test stimulus, opto_gen \
                      for optogenetic activation and neuron_flag to pinpoint which neuron to activate/deactivate
        stim_params : stimulus duration and values
        fixed_inp   : fixed input values for each neuron in plastic network

    returns:
        firing rates of neurons, weights of plastic network(if plastic else empty list)
    '''

    # Weights of network
    w_ed = weight_dict['wed']
    w_pe = weight_dict['wpe']
    w_pp = weight_dict['wpp']
    w_se = weight_dict['wse']
    w_ve = weight_dict['wve']
    w_de = weight_dict['wde']
    w_sv = weight_dict['wsv']
    w_vs = weight_dict['wvs']
    w_ep = weight_dict['wep']
    w_pv = weight_dict['wpv']
    w_ps = weight_dict['wps']
    w_ds = weight_dict['wds']

    # Number of neurons
    Ne = neurons_dict['Ne']
    Ni = neurons_dict['Ni']

    # timestep
    dt = 0.1

    # Number of connections between pre & post

    N_wep = int(np.round(Ni * 0.6))  # decimal numbers indicates the connection probability
    N_wed = int(np.round(Ne * 1))
    N_wde = int(np.round(Ne * 0.1))
    N_wds = int(np.round(Ni * 0.55))
    N_wpe = int(np.round(Ne * 0.45))
    N_wpp = int(np.round(Ni * 0.5))
    N_wps = int(np.round(Ni * 0.6))
    N_wpv = int(np.round(Ni * 0.5))
    N_wse = int(np.round(Ne * 0.35))
    N_wsv = int(np.round(Ni * 0.5))
    N_wve = int(np.round(Ne * 0.1))
    N_wvs = int(np.round(Ni * 0.45))

    # Create synapses
    W_EP, C_EP = create_synapse(w_ep, Ne, Ni, N_wep)
    W_ED, C_ED = np.identity(Ne) * w_ed, np.identity(Ne)
    W_DE, C_DE = create_synapse(w_de, Ne, Ne, N_wde)
    W_PE, C_PE = create_synapse(w_pe, Ni, Ne, N_wpe)
    W_PP, C_PP = create_synapse(w_pp, Ni, Ni, N_wpp)
    W_SE, C_SE = create_synapse(w_se, Ni, Ne, N_wse)
    W_SV, C_SV = create_synapse(w_sv, Ni, Ni, N_wsv)
    W_VE, C_VE = create_synapse(w_ve, Ni, Ne, N_wve)
    W_VS, C_VS = create_synapse(w_vs, Ni, Ni, N_wvs)
    W_PS, C_PS = create_synapse(w_ps, Ni, Ni, N_wps)
    W_PV, C_PV = create_synapse(w_pv, Ni, Ni, N_wpv)
    W_DS, C_DS = create_synapse(w_ds, Ne, Ni, N_wds)

    # Learning rates for plasticity
    η1 = eta_dict['η1']
    η2 = eta_dict['η2']
    η3 = eta_dict['η3']
    η4 = eta_dict['η4']

    # Stimulus on/off for neuron - params
    Ve = stim_dict['Ve']
    Vp = stim_dict['Vp']
    Mp = stim_dict['Mp']
    Vs = stim_dict['Vs']
    Ms = stim_dict['Ms']
    Vv = stim_dict['Vv']
    Mv = stim_dict['Mv']
    Md = stim_dict['Md']

    # stimulus duration and values
    stim_value = stim_params['stim_value']
    stim_count = int(stim_params['stim_dur'])  # Incase of training the plastic network
    no_visual = int(stim_params['no_vis_stim'])

    # Baseline firing rates
    re = rates_dict['re']
    rd = rates_dict['rd']
    rp = rates_dict['rp']
    rs = rates_dict['rs']
    rv = rates_dict['rv']

    # Flags
    test = flag_dict['test']
    opto_gen = flag_dict['opto_gen']
    neuron_flag = flag_dict['neuron_flag']
    opto_val = flag_dict['opto_val']    