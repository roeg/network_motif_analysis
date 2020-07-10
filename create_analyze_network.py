import numpy as np
from numpy.random import randn, rand, permutation
from scipy.stats import norm
from network.nw_creator import create_network
from network.nw import load_network
import matplotlib.pyplot as plt

n_neurons = (100, 100)
celltypes = 'e', 'i'

# combine degenerate triplets (i.e., use symmetries) to 16 total motifs
# for motif ID graphs, see Figure 6 here: https://www.frontiersin.org/articles/10.3389/fnana.2014.00129/full
motifs = dict()
motifs[1] = (0b111111,)
motifs[2] = (0b101010, 0b010101)
motifs[3] = (0b111110, 0b111101, 0b111011, 0b110111, 0b101111, 0b011111)
motifs[4] = (0b110110, 0b111001, 0b011110, 0b101101, 0b100111, 0b011011)
motifs[5] = (0b110101, 0b011101, 0b010111)
motifs[6] = (0b111010, 0b101110, 0b101011)
motifs[7] = (0b101001, 0b100110, 0b011010, 0b010110, 0b011001, 0b100101)

# double connected (i.e., two edges have at least one connection)
motifs[8] = (0b111100, 0b110011, 0b001111)
motifs[9] = (0b100100, 0b011000, 0b010001, 0b100010, 0b001001, 0b000110)
motifs[10] = (0b110100, 0b110001, 0b011100, 0b001101, 0b010011, 0b000111)
motifs[11] = (0b111000, 0b110010, 0b101100, 0b001110, 0b100011, 0b001011)
motifs[12] = (0b001010, 0b100001, 0b010100)
motifs[13] = (0b000101, 0b010010, 0b101000)

# singlet motifs (i.e., one edge has at least one connection)
motifs[14] = (0b110000, 0b001100, 0b000011)
motifs[15] = (0b100000, 0b010000, 0b001000, 0b000100, 0b000010, 0b000001)
motifs[16] = (0b000000,)

n_motifs = 0
for key, val in motifs.items():
    n_motifs += len(val)
assert n_motifs == 64


def distribute_neurons(n):
    # let's distribute neurons according to a 3D normal distribution
    sigma = 200.0
    return [sigma * randn(3) for i in range(n)]


def decide_connection_distance(neuron1, neuron2):
    # set up connectivity as another normal distribution, only depending on distance between neurons
    x1 = neuron1.location
    x2 = neuron2.location
    diff = np.sqrt(np.dot(x1 - x2, x1 - x2))
    scale = 100.0
    threshold = norm.pdf(diff, loc=0.0, scale=scale) * scale
    return rand() < threshold


def decide_connection_distance_type(neuron1, neuron2):
    # set up connectivity as another normal distribution, only depending on distance between neurons
    x1 = neuron1.location
    x2 = neuron2.location
    diff = np.sqrt(np.dot(x1 - x2, x1 - x2))
    if neuron1.celltype == 'i' or neuron2.celltype == 'i':
        scale = 100.0
        threshold = np.sinc(diff / scale) ** 2
    else:
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
    print('Computing spectrum for %d triplets' % len(triplets))
    # spectrum = np.zeros(64) # histogram of triplet occurrences
    spectrum = np.zeros(16) # histogram of degenerate triplet occurrences
    for t in triplets:
        triplet_code = 0
        t_ = list(t)
        triplet_code += connections_table.loc[t_[0], t_[1]] * 0b000001
        triplet_code += connections_table.loc[t_[1], t_[0]] * 0b000010
        triplet_code += connections_table.loc[t_[0], t_[2]] * 0b000100
        triplet_code += connections_table.loc[t_[2], t_[0]] * 0b001000
        triplet_code += connections_table.loc[t_[1], t_[2]] * 0b010000
        triplet_code += connections_table.loc[t_[2], t_[1]] * 0b100000
        # spectrum[int(triplet_code)] += 1.0
        for key, val in motifs.items():
            if triplet_code in val:
                spectrum[key - 1] += 1.0

    return spectrum


def create_save_network(out_folder):
    # create params dictionary
    params = dict()
    params['N'] = n_neurons
    params['celltypes'] = celltypes
    params['neuron_distribution'] = distribute_neurons
    # params['connection_pattern'] = decide_connection_distance
    params['connection_pattern'] = decide_connection_distance_type

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