import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from utils import Utils


def draw_graph(j):

    global step, numNeurons, neuralNetwork, neuronsUsage, utils
    print(step)

    if step > 0:

        for i in range(len(neuronsUsage)):
            neuronsUsage[i] -= 1

        for city in coordinates:
            winnerIdx = utils.index_of_winner_neuron(neuralNetwork, city)
            # Update winner neuron weights
            neuralNetwork[winnerIdx] = utils.update_neuron(neuralNetwork[winnerIdx], city)
            winnerAlready = neuronsUsage[winnerIdx] == 3
            neuronsUsage[winnerIdx] = 3  # If a neuron wins has 3 more chances of winning before been deleted

            if winnerAlready:  # If winner neuron has already won in this iteration
                if random.uniform(0, 1) < 0.5:
                    idx = (winnerIdx -1) % len(neuralNetwork)
                    newNeuron = utils.get_new_neuron(neuralNetwork[idx], neuralNetwork[winnerIdx])
                    neuralNetwork = np.insert(neuralNetwork, winnerIdx, newNeuron, axis=0)
                    neuronsUsage = np.insert(neuronsUsage, winnerIdx, 3)
                    winnerIdx += 1
                else:
                    idx = (winnerIdx +1) % len(neuralNetwork)
                    newNeuron = utils.get_new_neuron(neuralNetwork[idx], neuralNetwork[winnerIdx])
                    neuralNetwork = np.insert(neuralNetwork, winnerIdx +1, newNeuron, axis=0)
                    neuronsUsage = np.insert(neuronsUsage, winnerIdx +1, 3)

            # Update neuron neighbors weights
            for i in range(1, utils.radius + 1):
                idx = (winnerIdx - i) % len(neuralNetwork)
                neuralNetwork[idx] = utils.update_neuron(neuralNetwork[idx], city)
                idx = (winnerIdx + i) % len(neuralNetwork)
                neuralNetwork[idx] = utils.update_neuron(neuralNetwork[idx], city)

        utils.update_parameters()

        # Delete non winner neurons in 3 iterations
        toDelete = []
        for neuronIdx in range(len(neuronsUsage)):
            if neuronsUsage[neuronIdx] == 0:
                toDelete.append(neuronIdx)
        neuralNetwork = np.delete(neuralNetwork, toDelete, axis=0)
        neuronsUsage = np.delete(neuronsUsage, toDelete)

    numNeurons = len(neuralNetwork)
    print(numNeurons)

    neuralX, neuralY = neuralNetwork[:, 0], neuralNetwork[:, 1]
    neuralX, neuralY = np.append(neuralX, neuralX[0]), np.append(neuralY, neuralY[0])

    cities.cla()
    cities.plot(X, Y, 'go')
    neurons.plot(neuralX, neuralY, 'y.-')

    # if step == 10:
    #     animation.event_source.stop()
    # plt.savefig(str(step) + '.png')

    step += 1


# dataFile = 'Data_WesternSahara.txt'
# dataFile = 'Data_Djibouti.txt'
# dataFile = 'Data_Uruguay.txt'
dataFile = 'Data_Qatar.txt'


file = open(dataFile, 'rt')
dimension = 0
coordinates, X, Y = None, None, None

# Read city coordinates from text file
for line in file:
    words = line.split()
    if words[0] == 'DIMENSION' or words[0] == 'DIMENSION:':
        dimension = int(words[len(words) - 1])
        coordinates = np.empty(shape=(dimension, 2))
    if words[0].isdigit() and coordinates is not None:
        coordinates[int(words[0])-1] = [float(words[1]), float(words[2])]

X, Y = coordinates[:, 0], coordinates[:, 1]
maxX, minX, maxY, minY = np.amax(X), np.amin(X), np.amax(Y), np.amin(Y)

####### Generate ANN
numNeurons = dimension + dimension//2
neuralNetwork = np.empty(shape=(numNeurons, 2))
centerX, centerY = (minX+maxX)/2, (minY+maxY)/2
radiusX, radiusY = (maxX - centerX)/4, (maxY - centerY)/4
# Get neuron coordinates in a circle shape
for i in range(numNeurons):
    neuralNetwork[i][0] = centerX + math.cos((2*math.pi)*(i/numNeurons)) * radiusX
    neuralNetwork[i][1] = centerY + math.sin((2*math.pi)*(i/numNeurons)) * radiusY

neuronsUsage = np.full(shape=numNeurons, fill_value=3)

####### Generate figure
fig = plt.figure()
cities = fig.add_subplot(1, 1, 1)
neurons = fig.add_subplot(1, 1, 1)

step = 0

utils = Utils(numNeurons)

animation = ani.FuncAnimation(fig, draw_graph, interval=1000, cache_frame_data=False)

plt.xticks([])
plt.yticks([])
plt.show()
