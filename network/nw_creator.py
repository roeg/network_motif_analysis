import numpy as np
import pandas as pd
from network.nw import Neuron, Network

def create_network(params):
    """
    main function for network creation
    :param params: dictionary with entries 'N' - tuple with number of neurons per type; 'celltypes' - tuple with names
    of celltypes; 'neuron_distribution' - callable taking as input integer N and returning locations of N neurons;
    'connection_pattern' - callable taking as input location of two neurons (i.e., two 3-component vectors)
    :return: Network instance
    """
    # create all neurons
    neurons = []
    id_offset = 0
    for i, ct in enumerate(params['celltypes']):
        ct_number = params['N'][i]
        locations = params['neuron_distribution'](ct_number)
        neurons_ = [Neuron(ct, j + id_offset, locations[j]) for j in range(ct_number)]
        neurons.extend(neurons_)
        id_offset += len(neurons)

    # create connection matrix
    connections = np.zeros((len(neurons), len(neurons)))
    connection_pattern = params['connection_pattern']
    for i in range(len(neurons)):
        for j in range(len(neurons)):
            connections[i, j] = connection_pattern(neurons[i], neurons[j])

    return Network(neurons, connections)