import numpy as np
from numpy.random import randn, rand, permutation
from scipy.stats import norm
from network.nw_creator import create_network
from network.nw import load_network
import matplotlib.pyplot as plt

n_neurons = (100, 100)
celltypes = 'e', 'i'


def distribute_neurons(n):
    # let's distribute neurons according to a 3D normal distribution
    sigma = 200.0
    return [sigma * randn(3) for i in range(n)]


def decide_connection(x1, x2):
    # set up connectivity as another normal distribution, only depending on distance between neurons
    diff = np.sqrt(np.dot(x1 - x2, x1 - x2))
    scale = 100.0
    threshold = norm.pdf(diff, loc=0.0, scale=scale) * scale
    return rand() < threshold


def select_triplets(neuron_table, n_triplets):
    ids = neuron_table.index
    triplets = set()
    if n_triplets is None:
        print('Generating all possible triplets for %d neurons...' % len(ids))
        # select all possible triplets
        # careful - we don't check for memory requirements so this can easily get out of control (O(N^3))
        # also very inefficient implementation because we check each triplet against existing ones
        for id1 in ids:
            for id2 in ids:
                for id3 in ids:
                    if id2 == id1 or id3 == id1 or id3 == id2:
                        continue
                    triplets.add(frozenset([id1, id2, id3]))
        return triplets

    # randomly select a subset of triplets
    print('Generating %d randomly selected triplets for %d neurons...' % (n_triplets, len(ids)))
    while len(triplets) < n_triplets:
        # take first three elements of a permutation of indices
        triplets.add(frozenset(permutation(ids)[:3]))
    return triplets


def compute_motif_spectrum(triplets, connections_table):
    # encode each triplet as a number (0-63)
    # TODO: later combine degenerate triplets (i.e., use symmetries) - create LUT
    print('Computing spectrum for %d triplets' % len(triplets))
    spectrum = np.zeros(64) # histogram of triplet occurrences
    for t in triplets:
        triplet_code = 0
        t_ = list(t)
        triplet_code += connections_table.loc[t_[0], t_[1]] * 2 ** 0
        triplet_code += connections_table.loc[t_[1], t_[0]] * 2 ** 1
        triplet_code += connections_table.loc[t_[0], t_[2]] * 2 ** 2
        triplet_code += connections_table.loc[t_[2], t_[0]] * 2 ** 3
        triplet_code += connections_table.loc[t_[1], t_[2]] * 2 ** 4
        triplet_code += connections_table.loc[t_[2], t_[1]] * 2 ** 5
        spectrum[int(triplet_code)] += 1.0
    return spectrum


def create_save_network(out_folder):
    # create params dictionary
    params = dict()
    params['N'] = n_neurons
    params['celltypes'] = celltypes
    params['neuron_distribution'] = distribute_neurons
    params['connection_pattern'] = decide_connection

    network = create_network(params)
    # empirical connection probability
    pc = 1.0 * np.sum(network.connections) / (network.connections.shape[0] ** 2)
    print('Mean connection probability = %.2f' % pc)

    # Let's take a look at the network we just created
    # 2D projections (x-y and x-z) of the neuron locations
    # colored by cell type
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    for ct in celltypes:
        locations = []
        for neuron in network.neurons:
            if neuron.celltype == ct:
                locations.append(neuron.location)
        locations = np.array(locations)
        ax1.plot(locations[:, 0], locations[:, 1], 'o', label=ct)
        ax2.plot(locations[:, 0], locations[:, 2], 'o')
    plt.legend()
    plt.show()

    # Let's save this network for later analysis
    network.save_network(out_folder)


def analyze_saved_network(nw_folder, triplet_samples=None):
    nw = load_network(nw_folder)

    # select triplets from all cell types, and then from each cell type separately
    triplets_all = select_triplets(nw.neuron_table, triplet_samples)
    triplets_e = select_triplets(nw.neuron_table[nw.neuron_table["celltype"] == "e"], triplet_samples)
    triplets_i = select_triplets(nw.neuron_table[nw.neuron_table["celltype"] == "i"], triplet_samples)

    # compute motif spectrum for selections
    spectrum_all = compute_motif_spectrum(triplets_all, nw.connection_table)
    spectrum_e = compute_motif_spectrum(triplets_e, nw.connection_table)
    spectrum_i = compute_motif_spectrum(triplets_i, nw.connection_table)

    # plot spectra
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(spectrum_all))
    width = 0.2
    ax1.bar(x - 1.5 * width, np.log(spectrum_all + 1), label='All')
    ax1.bar(x, np.log(spectrum_e + 1), label='e')
    ax1.bar(x + 1.5 * width, np.log(spectrum_i + 1), label='i')
    ax1.set_xlabel('Motif ID')
    ax1.set_ylabel('Log frequency')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    folder = '/Users/robert/project_src/network_motif_analysis/data/network'
    n_triplet_samples = 10000
    create_save_network(folder)
    analyze_saved_network(folder, n_triplet_samples)