import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np

import os
import sys
from pathlib import Path

p = Path(os.getcwd())
#os.chdir(p.parent)
path = os.getcwd()

# plt.style.use('seaborn-white')


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 9

rc('axes', linewidth=1)


def plot_rates(re, rd, rp, rv, rs, stim_len):
    '''Function to plot the firing rates of all neurons'''

    fig, axs = plt.subplots(5, 1, figsize=(6, 4), dpi=150)
    fig.suptitle("PPE neurons for SOM = Motor & VIP = Visual")
    t = np.linspace(0, stim_len, len(re))
    rates = [re, rd, rp, rv, rs]
    labels = ["Exc", "  D", " PV", "VIP", "SOM"]
    colors = ["red", "orange", "green", "blue", "magenta"]
    for i, ax in enumerate(axs.ravel()):
        axs[i].plot(t, np.mean(rates[i], axis=1), label=labels[i], color=colors[i])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        # axs[i].spines['left'].set_visible(False)
        # axs[i].spines['bottom'].set_visible(True)
        axs[i].legend(loc='upper right', bbox_to_anchor=(1.04, 1.04), frameon=False)

        if i != len(axs) - 1:
            ax.set_xticks([])
            axs[i].spines['bottom'].set_visible(False)

    fig.text(0.5, 0.02, 'Time(s)', ha='center')
    fig.text(0.04, 0.5, 'Mean Firing rate(Hz)', va='center', rotation='vertical')
    plt.subplots_adjust(top=0.94)
    fig.savefig(path + '/Results/rates_plastic_network.png')
    # fig.tight_layout()


def plot_plastic_weights(wpv_track, wps_track, wep_track, wds_track, o_wpv, o_wps, o_wds, o_wep, stim_len):
    fig, axs = plt.subplots(4, 1, figsize=(6, 4), dpi=150)
    fig.suptitle("Plastic weights - PPE Neurons")
    t = np.linspace(0, stim_len, len(wpv_track))
    weights = [wep_track, wds_track, wpv_track, wps_track]
    labels = ["WEP", "WDS", "WPV", "WPS"]
    colors = ["red", "orange", "green", "blue"]
    optimal = [o_wep, o_wds, o_wpv, o_wps]
    for i, ax in enumerate(axs.ravel()):
        axs[i].plot(t, weights[i], label=labels[i], color=colors[i])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].axhline(y=optimal[i], xmin=0, color=colors[i], linestyle="dotted")
        # axs[i].spines['left'].set_visible(False)
        # axs[i].spines['bottom'].set_visible(False)
        axs[i].legend(loc="upper right", frameon=False)

        if i != len(axs) - 1:
            ax.set_xticks([])
            axs[i].spines['bottom'].set_visible(False)

    fig.text(0.5, 0.02, 'Time(s)', ha='center')
    fig.text(0.04, 0.5, 'Weight value', va='center', rotation='vertical')
    # fig.tight_layout()
    plt.subplots_adjust(top=0.94)
    fig.savefig(path + '/Results/weights_plastic_network.png')
    