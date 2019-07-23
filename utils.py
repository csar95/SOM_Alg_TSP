import math


class Utils:

    def __init__(self, numNeurons):
        self.learningRate = .2
        self.radius = int(numNeurons*0.1)
        self.newNeuronParam = .5

    @staticmethod
    def euclidean_distance(p, q):
        return math.sqrt((q[0] - p[0])**2 + (q[1] - p[1])**2)

    # @staticmethod
    def index_of_winner_neuron(self, aNN, city):
        index = 0
        minDistance = 9999999.0
        for i in range(len(aNN)):
            distance = self.euclidean_distance(aNN[i], city)
            if distance < minDistance:
                index = i
                minDistance = distance
        return index

    def update_neuron(self, neuron, city):
        return [neuron[0] + self.learningRate * (city[0] - neuron[0]),
                neuron[1] + self.learningRate * (city[1] - neuron[1])]

    def get_new_neuron(self, fellowNeuron, neuron):
        return [0.3 * fellowNeuron[0] + 0.7 * neuron[0],  # + 0.01 * self.newNeuronParam,
                0.3 * fellowNeuron[1] + 0.7 * neuron[1]]  #+ 0.01 * self.newNeuronParam]

    def update_parameters(self):
        self.learningRate *= 0.999
        self.radius = int(self.radius * 0.995)
