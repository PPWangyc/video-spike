import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import bits_per_spike
from sklearn.metrics import r2_score
import argparse
import torch
import matplotlib.patches as mpatches
# load rrr model
# rrr_result = torch.load('tmp')['RRRGD_model']
# eids = rrr_result['eids']
# print(eids)
# model = rrr_result['model']
# V_numpy = model['V'].detach().cpu().numpy()
# # Create a lineplot of the weights
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# ax.plot(V_numpy.flatten(), color='black')
# # add segmentation line for each rank
# for i in range(1, V_numpy.shape[0]):
#     ax.axvline(x=i*V_numpy.shape[1], color='red', linestyle='--')
# ax.set_xlabel('Weight Index', fontsize=15)
# ax.set_ylabel('Weight Value', fontsize=15)
# fig.savefig('rrr_model_weights_line.png')
# exit()

# ax.set_title(f'RRR Model Weights, EID: {eids[0][:5]}', fontsize=20)
# # Create a heatmap
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# im = ax.imshow(V_numpy, aspect='auto', cmap='binary')
# ax.set_title(f'RRR Model Weights, EID: {eids[0][:5]}', fontsize=20)
# ax.set_xlabel('Time Dimension', fontsize=15)
# ax.set_ylabel('Rank', fontsize=15)
# fig.colorbar(im, ax=ax)
# fig.savefig('rrr_model_weights.png')
# plt.close(fig)
# exit()
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
# for idx, eid in enumerate(eids):
me_result_all = np.load(f'me-all_result.npy', allow_pickle=True).item()
of_result_all = np.load(f'of-all_result.npy', allow_pickle=True).item()
all_result_all = np.load(f'all_result.npy', allow_pickle=True).item()
input_of_data = np.load(f'data/data_rrr_of-all.npy', allow_pickle=True).item()
input_data_me = np.load(f'data/data_rrr_all.npy', allow_pickle=True).item()

# visualize me raster plot
for idx, eid in enumerate(eids):
    me_result = input_data_me[eid]['X'][1]
    of_result = input_of_data[eid]['X'][1]
    # Create a raster plot for ME
    fig, axs = plt.subplots(3, 1, figsize=(4, 10))
    im_me = axs[0].imshow(me_result[...,0], aspect='auto', cmap='binary')
    axs[0].set_title(f'EID: {eid[:5]}, ME Raster Plot')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Trial IDX')
    fig.colorbar(im_me, ax=axs[0], orientation='vertical')
    print(of_result.shape)
    exit()
    im_of_x = axs[1].imshow(of_result[...,0], aspect='auto', cmap='binary')
    axs[1].set_title(f'EID: {eid[:5]}, OF-X Raster Plot')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Trial IDX')
    fig.colorbar(im_of_x, ax=axs[1], orientation='vertical')
    # Create a raster plot for OF
    im_of_y = axs[2].imshow(of_result[...,1], aspect='auto', cmap='binary')
    axs[2].set_title(f'EID: {eid[:5]}, OF-Y Raster Plot')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Trial IDX')
    fig.colorbar(im_of_y, ax=axs[2], orientation='vertical')
    # hspace to adjust the space between subplots
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(f'{eid[:5]}_raster_plot.png')
exit()
eid_result = {}
for eid in me_result_all:
    print(f"Processing {eid}")
    me_result = me_result_all[eid]
    of_result = of_result_all[eid]
    all_result = all_result_all[eid]

    gt = me_result['gt']
    me_pred = me_result['pred']
    of_pred = of_result['pred']
    all_pred = all_result['pred']

    # calculate population bps 
    me_bps = np.nanmean(me_result['co_bps'])
    of_bps = np.nanmean(of_result['co_bps'])
    all_bps = np.nanmean(all_result['co_bps'])

    eid_result[eid] = {'me_bps': me_bps, 'of_bps': of_bps, 'all_bps': all_bps}

    # calculate population r2
    me_r2 = r2_score(gt.flatten(), me_pred.flatten())
    of_r2 = r2_score(gt.flatten(), of_pred.flatten())
    all_r2 = r2_score(gt.flatten(), all_pred.flatten())

    # neuron-wise r2
    me_r2_neuron = np.array([r2_score(gt[..., i].flatten(), me_pred[..., i].flatten()) for i in range(gt.shape[-1])])
    of_r2_neuron = np.array([r2_score(gt[..., i].flatten(), of_pred[..., i].flatten()) for i in range(gt.shape[-1])])
    all_r2_neuron = np.array([r2_score(gt[..., i].flatten(), all_pred[..., i].flatten()) for i in range(gt.shape[-1])])

    # block
    block = input_data_me[eid]['X'][1][:,0, -1]
    choice = input_data_me[eid]['X'][1][:,0, -2]
    # unique block and choice
    unique_block = np.unique(block)
    unique_choice = np.unique(choice)
    # get idxs for each block and choice
    block_idxs = [np.where(block == b)[0] for b in unique_block]
    choice_idxs = [np.where(choice == c)[0] for c in unique_choice]
    # get idxs where the block and choice are the same
    choice_block_idxs = [np.intersect1d(c,b) for c in choice_idxs for b in block_idxs]
    # choice_block_idxs = choice_idxs
    # get labels for each bloc k and choice
    # only keep 1 decimal points
    block_labels = [f'C: {c:.0f}, B: {b:.1f}' for c in unique_choice for b in unique_block]
    # block_labels = [f'C: {c:.1f}' for c in unique_choice]
    # Determine the number of plots (2 for scatter + 10 neurons)
    fig, axs = plt.subplots(12, 1, figsize=(15, 40))  # 10 neurons + 2 scatter plots, all in one column
    # disable axs ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    min_bps = min(np.min(me_result['co_bps']), np.min(of_result['co_bps']))
    max_bps = max(np.max(me_result['co_bps']), np.max(of_result['co_bps']))
    min_r2 = min(np.min(me_r2_neuron), np.min(of_r2_neuron))
    max_r2 = max(np.max(me_r2_neuron), np.max(of_r2_neuron))
    # R2 Scatter Plot, use entire row
    ax_r2 = plt.subplot2grid((12, 3), (0, 0), colspan=3, fig=fig)
    ax_r2.scatter(me_r2_neuron, of_r2_neuron)
    ax_r2.set_xlim([min_r2, max_r2])
    ax_r2.set_ylim([min_r2, max_r2])
    ax_r2.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', lw=2, color='red')
    ax_r2.set_xlabel('ME R2')
    ax_r2.set_ylabel(f'{input_mod} R2')
    ax_r2.set_title(f'EID: {eid[:5]} \n R2 Scatter, ME R2: {me_r2:.3f}, {input_mod} R2: {of_r2:.3f}')

    # BPS Scatter Plot, use entire row
    ax_bps = plt.subplot2grid((12, 3), (1, 0), colspan=3, fig=fig)
    ax_bps.scatter(me_result['co_bps'], of_result['co_bps'])
    ax_bps.set_xlim([min_bps, max_bps])
    ax_bps.set_ylim([min_bps, max_bps])
    ax_bps.plot([min_bps, max_bps], [min_bps, max_bps], 'k--', lw=2, color='red')
    ax_bps.set_xlabel('ME BPS')
    ax_bps.set_ylabel(f'{input_mod} BPS')
    ax_bps.set_title(f'BPS Scatter, ME BPS: {me_bps:.3f}, {input_mod} BPS: {of_bps:.3f}')

    # Top 10 active neurons plots
    # top_neuron_idx = np.argsort(np.mean(of_result['gt'], axis=(0, 1)))[::-1][:10]
    top_neuron_idx = np.argsort(all_r2_neuron)[::-1][:10]
    for j, neuron_idx in enumerate(top_neuron_idx):
        row = j + 2  # Start from the 3rd row
        gt_neuron = me_result['gt'][..., neuron_idx]
        me_neuron = me_result['pred'][..., neuron_idx]
        of_neuron = of_result['pred'][..., neuron_idx]
        all_neuron = all_result['pred'][..., neuron_idx]
        # flatten idx
        _choice_block_idxs = np.concatenate(choice_block_idxs)
        # sort gt based on block and choice
        gt_neuron = gt_neuron[_choice_block_idxs]
        me_neuron = me_neuron[_choice_block_idxs]
        of_neuron = of_neuron[_choice_block_idxs]
        all_neuron = all_neuron[_choice_block_idxs]
        # Create three subplots for GT, ME, OF in the same row with a color bar for each
        axs_gt = plt.subplot2grid((12, 4), (row, 0), fig=fig)
        axs_me = plt.subplot2grid((12, 4), (row, 1), fig=fig)
        axs_of = plt.subplot2grid((12, 4), (row, 2), fig=fig)
        axs_all = plt.subplot2grid((12, 4), (row, 3), fig=fig)
        
        # add vertical line on the im_gt side to sperate block labels
        ymin = 0
        colormap = plt.cm.get_cmap('tab20', len(block_labels))
        for i in range(len(block_labels)):
            length = choice_block_idxs[i].shape[0]
            ymax = ymin + length
            axs_gt.plot([0, 0], [ymin, ymax], color=colormap(i), linewidth=10)
            # add label for each block
            # axs_gt.text(-60, ymin, block_labels[i], fontsize=10, color=colormap(i))
            axs_me.plot([0, 0], [ymin, ymax], color=colormap(i), linewidth=10)
            axs_of.plot([0, 0], [ymin, ymax], color=colormap(i), linewidth=10)
            axs_all.plot([0, 0], [ymin, ymax], color=colormap(i), linewidth=10)
            ymin += length
        # add legend for block labels
        im_gt = axs_gt.imshow(gt_neuron, aspect='auto', cmap='binary')
        im_me = axs_me.imshow(me_neuron, aspect='auto', cmap='binary')
        im_of = axs_of.imshow(of_neuron, aspect='auto', cmap='binary')
        im_all = axs_all.imshow(all_neuron, aspect='auto', cmap='binary')

        axs_gt.set_title(f'Neuron idx: {neuron_idx}, GT')
        axs_me.set_title(f'ME, R2: {me_r2_neuron[neuron_idx]:.3f}, CO-BPS: {me_result["co_bps"][neuron_idx]:.3f}')
        axs_of.set_title(f'{input_mod}, R2: {of_r2_neuron[neuron_idx]:.3f}, CO-BPS: {of_result["co_bps"][neuron_idx]:.3f}')
        axs_all.set_title(f'All, R2: {all_r2:.3f}, CO-BPS: {all_bps:.3f}')

        # set x,y lim for all subplots
        axs_gt.set_xlim([0, gt_neuron.shape[1]])
        axs_gt.set_ylim([0, gt_neuron.shape[0]])
        axs_me.set_xlim([0, me_neuron.shape[1]])
        axs_me.set_ylim([0, me_neuron.shape[0]])
        axs_of.set_xlim([0, of_neuron.shape[1]])
        axs_of.set_ylim([0, of_neuron.shape[0]])
        axs_all.set_xlim([0, all_neuron.shape[1]])
        axs_all.set_ylim([0, all_neuron.shape[0]])
        # set legend for block labels
        legend_handles = []
        for i, label in enumerate(block_labels):
            # Create a patch for each label with the corresponding color
            patch = mpatches.Patch(color=colormap(i), label=label)
            legend_handles.append(patch)
        # reverse the legend handles
        legend_handles = legend_handles[::-1]
        axs_gt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-.7, 1))
        # set y label for the first column
        axs_gt.set_ylabel('Trial IDX')
        axs_gt.set_xlabel('Time')
        axs_me.set_xlabel('Time')
        axs_of.set_xlabel('Time')
        axs_all.set_xlabel('Time')

        # set horizontal colorbar for each subplot
        # only show min and max value
        fig.colorbar(im_gt, ax=axs_gt, orientation='horizontal', pad=.3)
        fig.colorbar(im_me, ax=axs_me, orientation='horizontal', pad=.3)
        fig.colorbar(im_of, ax=axs_of, orientation='horizontal', pad=.3)
        fig.colorbar(im_all, ax=axs_all, orientation='horizontal', pad=.3)
        
    # Adjust spacing and save
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    fig.savefig(f'{eid[:5]}_neuron_plots.png')
    plt.close(fig)  # Close figure to save memory

# # Plotting a boxplot for ME, OF, and All BPS
# based on eid_result
me_bps = [eid_result[eid]['me_bps'] for eid in eid_result]
of_bps = [eid_result[eid]['of_bps'] for eid in eid_result]
all_bps = [eid_result[eid]['all_bps'] for eid in eid_result]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
print(me_bps)
print(of_bps)
print(all_bps)
ax.boxplot([me_bps, of_bps, all_bps], labels=['ME', 'OF-2D', 'All'], medianprops=dict(color="black"))
# plot the bar plot to show the mean value
ax.bar(1, np.mean(me_bps),  label='ME',width=0.18)
ax.bar(2, np.mean(of_bps), label='OF-2D',width=0.18)
ax.bar(3, np.mean(all_bps), label='All',width=0.18)
# add the mean value on top of the bar
ax.text(1, np.mean(me_bps), f'{np.mean(me_bps):.3f}', ha='center', va='bottom')
ax.text(2, np.mean(of_bps), f'{np.mean(of_bps):.3f}', ha='center', va='bottom')
ax.text(3, np.mean(all_bps), f'{np.mean(all_bps):.3f}', ha='center', va='bottom')
ax.set_ylabel('BPS')
fig.savefig('bps_boxplot.png')
plt.close(fig)


