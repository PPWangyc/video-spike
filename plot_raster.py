import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import bits_per_spike
from sklearn.metrics import r2_score 
import argparse

# Argument parsing setup
parser = argparse.ArgumentParser(description='Plot scatter plots for ME and OF')
parser.add_argument('--input_mod', type=str, default='of-2d', help='Input modality')
args = parser.parse_args()

# Reading experiment IDs from file
eids = []
with open('data/eid.txt') as f:
    for line in f:
        eids.append(line.strip())

input_mod = args.input_mod
num_sessions = len(eids)  # Number of subplots needed

# Loop through each session to create individual figures
for idx, eid in enumerate(eids):
    me_result = np.load(f'{eid[:5]}_me_result.npy', allow_pickle=True).item()
    of_result = np.load(f'{eid[:5]}_{input_mod}_result.npy', allow_pickle=True).item()

    # Create a main figure with 12 subplots (2 for R2 and BPS, and 10 for top neurons)
    fig, axs = plt.subplots(12, 1, figsize=(10, 30), gridspec_kw={'height_ratios': [1, 1] + [1]*10})
    
    # R2 Scatter Plot
    axs[0].scatter(me_result['r2'], of_result['r2'])
    axs[0].set_xlabel('ME R2')
    axs[0].set_ylabel(f'{input_mod} R2')
    axs[0].set_title(f'Scatter R2: Session {eid[:5]}')
    axs[0].plot([0, 1], [0, 1], transform=axs[0].transAxes, color='red')  # Line for diagonal

    # BPS Scatter Plot
    axs[1].scatter(me_result['co_bps'], of_result['co_bps'])
    axs[1].set_xlabel('ME BPS')
    axs[1].set_ylabel(f'{input_mod} BPS')
    axs[1].set_title(f'Scatter BPS: Session {eid[:5]}')
    axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, color='red')  # Line for diagonal

    # Top 10 active neurons plots
    top_neuron_idx = np.argsort(np.mean(of_result['gt'], axis=(0,1)))[::-1][:10]
    for j, neuron_idx in enumerate(top_neuron_idx):
        gt_neuron = me_result['gt'][...,neuron_idx]
        me_neuron = me_result['pred'][...,neuron_idx]
        of_neuron = of_result['pred'][...,neuron_idx]
        print(j+2)
        axs[j+2].imshow(np.hstack([gt_neuron, me_neuron, of_neuron]), aspect='auto', cmap='binary')
        axs[j+2].set_title(f'Neuron {neuron_idx} GT | ME | {input_mod}')
        axs[j+2].set_xlabel('Time')
        axs[j+2].set_ylabel('Trial')
    fig.tight_layout()
    fig.savefig(f'{eid[:5]}_session_figure.png')
    plt.close(fig)  # Close the figure to save memory
