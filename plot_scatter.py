import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import bits_per_spike
from sklearn.metrics import r2_score 
import argparse

parser = argparse.ArgumentParser(description='Plot scatter plots for ME and OF')
parser.add_argument('--input_mod', type=str, default='of-2d', help='Input modality')
args = parser.parse_args()
eids = []
with open('data/eid.txt') as f:
    for line in f:
        eids.append(line.strip())

input_mod = args.input_mod
# Determine number of subplots needed
num_sessions = len(eids)

# Create figures for R2 and BPS with appropriate number of subplots
fig_r2, axs_r2 = plt.subplots(1, num_sessions, figsize=(5*num_sessions, 5))
fig_bps, axs_bps = plt.subplots(1, num_sessions, figsize=(5*num_sessions, 5))

for idx, eid in enumerate(eids):
    me_result = np.load(f'{eid[:5]}_me_result.npy', allow_pickle=True).item()
    of_result = np.load(f'{eid[:5]}_{input_mod}_result.npy', allow_pickle=True).item()

    me_neuron_r2 = np.array(me_result['r2'])
    of_neuron_r2 = np.array(of_result['r2'])

    of_gt = np.array(of_result['gt'])
    of_pred = np.array(of_result['pred'])

    op_population_bps = bits_per_spike(of_pred, of_gt)

    me_gt = np.array(me_result['gt'])
    me_pred = np.array(me_result['pred'])

    me_population_bps = bits_per_spike(me_pred, me_gt)

    me_r2 = np.nanmean([r2_score(me_gt[i], me_pred[i]) for i in range(len(me_gt))])
    of_r2 = np.nanmean([r2_score(of_gt[i], of_pred[i]) for i in range(len(of_gt))])

    min_r2 = np.min([np.min(me_neuron_r2), np.min(of_neuron_r2)])
    max_r2 = np.max([np.max(me_neuron_r2), np.max(of_neuron_r2)])
    # Plot R2 scatter
    axs_r2[idx].scatter(me_neuron_r2, of_neuron_r2)
    axs_r2[idx].set_xlabel('ME R2')
    axs_r2[idx].set_ylabel(f'{input_mod} R2')
    axs_r2[idx].plot([min_r2, max_r2], [min_r2, max_r2], color='red')  # Line for diagonal
    axs_r2[idx].legend([f'Session {eid[:5]} Neurons'])
    axs_r2[idx].set_title(f'ME ({me_r2:.3f}) vs {input_mod} ({of_r2:.3f})')

    me_neuron_bps = np.array(me_result['co_bps'])
    of_neuron_bps = np.array(of_result['co_bps'])
    min_bps = np.min([np.min(me_neuron_bps), np.min(of_neuron_bps)])
    max_bps = np.max([np.max(me_neuron_bps), np.max(of_neuron_bps)])
    # Plot BPS scatter
    axs_bps[idx].scatter(me_neuron_bps, of_neuron_bps)
    axs_bps[idx].set_xlabel('ME BPS')
    axs_bps[idx].set_ylabel(f'{input_mod} BPS')
    axs_bps[idx].plot([min_bps, max_bps], [min_bps, max_bps], color='red')  # Line for diagonal
    axs_bps[idx].legend([f'Session {eid[:5]} Neurons'])
    axs_bps[idx].set_title(f'ME ({me_population_bps:.3f} BPS) vs {input_mod} ({op_population_bps:.3f} BPS)')

# Save figures
fig_r2.tight_layout()
fig_r2.savefig('scatter_r2_sessions.png')
fig_bps.tight_layout()
fig_bps.savefig('scatter_bps_sessions.png')

plt.show()
