

class Neuron(object):
    def __init__(self, celltype, id, location):
        self.celltype = celltype
        self.id = id
        self.location = location


class Network(object):
    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections

    def save_network(self):
        # TODO: implement
        pass

    def load_network(self):
        # TODO: implement
        pass