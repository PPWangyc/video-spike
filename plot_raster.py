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

    # Determine the number of plots (2 for scatter + 10 neurons)
    fig, axs = plt.subplots(12, 1, figsize=(15, 40))  # 10 neurons + 2 scatter plots, all in one column
    # disable axs ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    min_bps = min(np.min(me_result['co_bps']), np.min(of_result['co_bps']))
    max_bps = max(np.max(me_result['co_bps']), np.max(of_result['co_bps']))
    min_r2 = min(np.min(me_result['r2']), np.min(of_result['r2']))
    max_r2 = max(np.max(me_result['r2']), np.max(of_result['r2']))
    # R2 Scatter Plot, use entire row
    ax_r2 = plt.subplot2grid((12, 3), (0, 0), colspan=3, fig=fig)
    ax_r2.scatter(me_result['r2'], of_result['r2'])
    ax_r2.set_xlim([min_r2, max_r2])
    ax_r2.set_ylim([min_r2, max_r2])
    ax_r2.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', lw=2)
    ax_r2.set_xlabel('ME R2')
    ax_r2.set_ylabel(f'{input_mod} R2')
    ax_r2.set_title('Scatter R2')

    # BPS Scatter Plot, use entire row
    ax_bps = plt.subplot2grid((12, 3), (1, 0), colspan=3, fig=fig)
    ax_bps.scatter(me_result['co_bps'], of_result['co_bps'])
    ax_bps.set_xlim([min_bps, max_bps])
    ax_bps.set_ylim([min_bps, max_bps])
    ax_bps.plot([min_bps, max_bps], [min_bps, max_bps], 'k--', lw=2)
    ax_bps.set_xlabel('ME BPS')
    ax_bps.set_ylabel(f'{input_mod} BPS')
    ax_bps.set_title('Scatter BPS')

    # Top 10 active neurons plots
    top_neuron_idx = np.argsort(np.mean(of_result['gt'], axis=(0, 1)))[::-1][:10]
    for j, neuron_idx in enumerate(top_neuron_idx):
        row = j + 2  # Start from the 3rd row
        gt_neuron = me_result['gt'][..., neuron_idx]
        me_neuron = me_result['pred'][..., neuron_idx]
        of_neuron = of_result['pred'][..., neuron_idx]

        # Create three subplots for GT, ME, OF in the same row with a color bar for each
        axs_gt = plt.subplot2grid((12, 3), (row, 0), fig=fig)
        axs_me = plt.subplot2grid((12, 3), (row, 1), fig=fig)
        axs_of = plt.subplot2grid((12, 3), (row, 2), fig=fig)

        im_gt = axs_gt.imshow(gt_neuron, aspect='auto', cmap='binary')
        im_me = axs_me.imshow(me_neuron, aspect='auto', cmap='binary')
        im_of = axs_of.imshow(of_neuron, aspect='auto', cmap='binary')

        axs_gt.set_title(f'Neuron {neuron_idx} GT')
        axs_me.set_title('ME')
        axs_of.set_title(f'{input_mod}')

        # figure out
        


    # Adjust spacing and save
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    fig.savefig(f'{eid[:5]}_neuron_plots.png')
    plt.close(fig)  # Close figure to save memory
