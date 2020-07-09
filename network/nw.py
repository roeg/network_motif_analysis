import numpy as np
import pandas as pd


def load_network(nw_folder):
    # load the neuron_table and connection_table
    neuron_name = nw_folder + '_neuron_table.h5'
    connection_name = nw_folder + '_connection_table.h5'
    neuron_table = pd.read_hdf(neuron_name, 'table')
    connection_table = pd.read_hdf(connection_name, 'table')
    nw = Network([], [])
    nw.neuron_table = neuron_table
    nw.connection_table = connection_table

    # here, we set up the neurons from the neuron_table
    neurons = []
    for index in nw.neuron_table.index:
        neuron_id = index
        celltype = nw.neuron_table.loc[index, 'celltype']
        loc = nw.neuron_table.loc[index, ['location_x', 'location_y', 'location_z']].to_numpy()
        neurons.append(Neuron(celltype, neuron_id, loc))
    nw.neurons = neurons
    return nw


class Neuron(object):
    def __init__(self, celltype, id, location):
        self.celltype = celltype
        self.id = id
        self.location = location


class Network(object):
    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections
        self.neuron_table = None
        self.connection_table = None

    def _set_up_df(self):
        # set up pandas data structures automatically
        celltypes = []
        ids = []
        locations = []
        for neuron in self.neurons:
            celltypes.append(neuron.celltype)
            ids.append(neuron.id)
            locations.append(neuron.location)
        locations = np.array(locations)
        self.neuron_table = pd.DataFrame({
            'id': ids,
            'celltype': celltypes,
            'location_x': locations[:, 0],
            'location_y': locations[:, 1],
            'location_z': locations[:, 2]
        })
        self.connection_table = pd.DataFrame(self.connections, index=ids, columns=ids)

    def save_network(self, out_name):
        # save the neuron_table and connection_table
        if self.neuron_table is None or self.connection_table is None:
            self._set_up_df()
        neuron_name = out_name + '_neuron_table.h5'
        connection_name = out_name + '_connection_table.h5'
        print('Saving neuron table as %s' % neuron_name)
        self.neuron_table.to_hdf(neuron_name, 'table')
        print('Saving connection table as %s' % connection_name)
        self.connection_table.to_hdf(connection_name, 'table')