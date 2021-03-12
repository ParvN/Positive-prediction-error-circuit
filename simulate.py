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

    if test == 1:
        V, M = test_stimulus(stim_value, dt)  # Test stimulus
        xE = fixed_inp['xe']
        xD = fixed_inp['xd']
        xP = fixed_inp['xp']
        xS = fixed_inp['xs']
        xV = fixed_inp['xv']
        if opto_gen == 1:  # Optogenetic act/inact of specific neuron
            if neuron_flag == 0:  # PV
                xP += opto_val
            elif neuron_flag == 1:  # SOM
                xS += opto_val
            elif neuron_flag == 2:  # VIP
                xV += opto_val
    else:  # If training
        M, V = train_stimulus(stim_value, stim_count, dt, no_visual)  # train stimulus

        xE = fixed_inp['xe']
        xD = fixed_inp['xd']
        xP = fixed_inp['xp']
        xS = fixed_inp['xs']
        xV = fixed_inp['xv']

    x_E = np.ones((len(V))) * xE  # Fixed input for all neurons
    x_D = np.ones((len(V))) * xD
    x_P = np.ones((len(V))) * xP
    x_S = np.ones((len(V))) * xS
    x_V = np.ones((len(V))) * xV

    stim_E = Ve * V + x_E  # Stimulus for all neurons
    stim_D = Md * M + x_D
    stim_P = Vp * V + Mp * M + x_P
    stim_S = Vs * V + Ms * M + x_S
    stim_V = Vv * V + Mv * M + x_V

    tauE = 60  # Time constants
    tauI = 2

    rE = []
    rD = []
    rP = []
    rS = []
    rV = []

    r_E = r_D = np.zeros((Ne))
    r_P = r_S = r_V = np.zeros((Ni))

    wep_track = []
    wds_track = []
    wpv_track = []
    wps_track = []

    I_e = []

    for idx in range(len(V)):

        dr_E1 = (1 / tauE) * (-r_E - (W_EP @ r_P) + (W_ED @ r_D) + stim_E[idx])
        dr_D1 = (1 / tauE) * (-r_D - (W_DS @ r_S) + (W_DE @ r_E) + stim_D[idx])
        dr_P1 = (1 / tauI) * (-r_P + (W_PE @ r_E) - (W_PP @ r_P) - (W_PS @ r_S) - (W_PV @ r_V) + stim_P[idx])
        dr_S1 = (1 / tauI) * (-r_S + (W_SE @ r_E) - (W_SV @ r_V) + stim_S[idx])
        dr_V1 = (1 / tauI) * (-r_V + (W_VE @ r_E) - (W_VS @ r_S) + stim_V[idx])

        r_E += dr_E1 * dt
        r_D += dr_D1 * dt
        r_P += dr_P1 * dt
        r_S += dr_S1 * dt
        r_V += dr_V1 * dt

        r_E = np.maximum(r_E, 0)
        r_D = np.maximum(r_D, 0)
        r_P = np.maximum(r_P, 0)
        r_S = np.maximum(r_S, 0)
        r_V = np.maximum(r_V, 0)

        if idx % 50 == 0:
            rE.append(r_E)
            rD.append(r_D)
            rP.append(r_P)
            rS.append(r_S)
            rV.append(r_V)
            I_e.append(((W_ED @ r_D) + stim_E[idx] - (W_EP @ r_P)))

        if test == 0:

            for i in range(C_EP.shape[0]):
                for j in range(C_EP.shape[1]):
                    Δw = η1 * (r_E[i] - re) * r_P[j]
                    W_EP[i, j] += Δw * C_EP[i, j]

            Є = 0.01
            for i in range(C_DS.shape[0]):
                for j in range(C_DS.shape[1]):
                    Δw = η2 * (r_D[i] - Є) * r_S[j]
                    W_DS[i, j] += Δw * C_DS[i, j]

            for i in range(C_PV.shape[0]):
                for j in range(C_PV.shape[1]):
                    Nei = 0
                    Δwij = 0
                    for k in range(C_EP.shape[0]):
                        if C_EP[k, i] == 1:
                            Nei += 1
                            Δwij += (re - r_E[k]) * r_V[j]
                    W_PV[i, j] += (η3 / Nei) * Δwij * C_PV[i, j]

            for i in range(C_PS.shape[0]):
                for j in range(C_PS.shape[1]):
                    Nei = 0
                    Δwij = 0
                    for k in range(C_EP.shape[0]):
                        if C_EP[k, i] == 1:
                            Nei += 1
                            Δwij += (re - r_E[k]) * r_S[j]
                    W_PS[i, j] += (η4 / Nei) * Δwij * C_PS[i, j]

            if idx % 50 == 0:
                wep_track.append(np.sum(W_EP[0]))
                wpv_track.append(np.sum(W_PV[0]))
                wps_track.append(np.sum(W_PS[0]))
                wds_track.append(np.sum(W_DS[0]))
    rE  = arr_conv(rE)
    rD  = arr_conv(rD)
    rP  = arr_conv(rP)
    rS  = arr_conv(rS)
    rV  = arr_conv(rV)
    I_e = arr_conv(I_e)

    rates['re'] = rE
    rates['rd'] = rD
    rates['rp'] = rP
    rates['rs'] = rS
    rates['rv'] = rV
    rates['Ie'] = I_e

    weights['w_ep'] = np.array(wep_track)
    weights['w_ds'] = np.array(wds_track)
    weights['w_pv'] = np.array(wpv_track)
    weights['w_ps'] = np.array(wps_track)

    return rates, weights
