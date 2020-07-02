import numpy as np
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
    for i, ct in enumerate(params['celltype']):
        ct_number = params['N'][i]
        locations = params['neuron_distribution'](ct_number)
        neurons_ = [Neuron(ct, j + id_offset, locations[j]) for j in range(ct_number)]
        neurons.extend(neurons_)
        id_offset += len(neurons)

    # create connection matrix
    # set up all pairwise locations
    locations = [n.location for n in neurons]
    connections = np.zeros((len(locations), len(locations)))
    connection_pattern = params['connection_pattern']
    for i in range(len(locations)):
        for j in range(len(locations)):
            connections[i, j] = connection_pattern(locations[i], locations[j])

    return Network(neurons, connections)