import numpy as np
import matplotlib.pyplot as plt
from utils.metric_utils import bits_per_spike

eids = []
with open('data/eid.txt') as f:
    for line in f:
        eids.append(line.strip())

input_mod = 'of-2d'
for eid in eids:
    me_result = np.load(f'{eid[:5]}_me_result.npy', allow_pickle=True).item()
    of_result = np.load(f'{eid[:5]}_{input_mod}_result.npy', allow_pickle=True).item()

    me_neuron_r2 = me_result['r2']
    of_neuron_r2 = of_result['r2']
    eid = of_result['eid']

    me_neuron_r2 = np.array(me_neuron_r2)
    of_neuron_r2 = np.array(of_neuron_r2)

    of_gt = np.array(of_result['gt'])
    of_pred = np.array(of_result['pred'])

    op_population_bps = bits_per_spike(of_pred, of_gt)

    me_gt = np.array(me_result['gt'])
    me_pred = np.array(me_result['pred'])

    me_population_bps = bits_per_spike(me_pred, me_gt)

    print(me_neuron_r2.shape)
    print(of_neuron_r2.shape)
    plt.figure()
    plt.scatter(me_neuron_r2, of_neuron_r2)
    plt.xlabel('ME R2')
    plt.ylabel(f'{input_mod} R2')
    # plot a line to show the diagonal
    # min and max of the two r2 values
    min_r2 = min(np.min(me_neuron_r2), np.min(of_neuron_r2))
    max_r2 = max(np.max(me_neuron_r2), np.max(of_neuron_r2))
    plt.plot([min_r2, max_r2], [min_r2, max_r2], color='red')
    # legend to show the number of neurons
    plt.legend([f'Session {eid[:5]} Neurons'])
    plt.title(f'ME vs {input_mod} R2')
    plt.savefig(f'scatter_r2_{eid[:5]}_{input_mod}.png')

    me_neuron_bps = me_result['co_bps']
    of_neuron_bps = of_result['co_bps']

    me_neuron_bps = np.array(me_neuron_bps)
    of_neuron_bps = np.array(of_neuron_bps)

    plt.figure()
    plt.scatter(me_neuron_bps, of_neuron_bps)
    plt.xlabel('ME BPS')
    plt.ylabel(f'{input_mod} BPS')
    # plot a line to show the diagonal
    # min and max of the two r2 values
    min_bps = min(np.min(me_neuron_bps), np.min(of_neuron_bps))
    max_bps = max(np.max(me_neuron_bps), np.max(of_neuron_bps))
    plt.plot([min_bps, max_bps], [min_bps, max_bps], color='red')
    plt.legend([f'Session {eid[:5]} Neurons'])
    plt.title(f'ME ({me_population_bps:.3f} BPS) vs {input_mod} ({op_population_bps:.3f} BPS)')
    plt.savefig(f'scatter_bps_{eid[:5]}_{input_mod}.png')
