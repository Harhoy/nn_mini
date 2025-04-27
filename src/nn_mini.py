
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

###########################################
# Different vectorizations and functions
###########################################

#loss function where output in a sigmoid (y_pred is return from sigmoid function)
def loss(y_pred, y):
    return (y_pred - y) * y_pred * (1 - y_pred)

#sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + 2.71 ** (-x))

def brier(y_pred, y):
    return (y_pred - y)**2

#vectorizations
loss_vectorized = np.vectorize(loss)
sigmoid_vectorized = np.vectorize(sigmoid)
brier_vectorized = np.vectorize(brier)

def convert2numpy(imgFile):
    return np.asarray(Image.open(imgFile))


def flatten(a):
    x, y = a.shape
    flat = []
    for i in range(x):
        for j in range(y):
            flat.append(a[i][j])
    return np.array(flat)



###########################################
# Classes
###########################################

class ImageDataLoader:

    def __init__(self, folder, items):

        #folder where subfolder for each category is placed
        self._folder = folder

        #data to be returned
        self._data = []

        #labels
        self._labels = []

        #number of categories to classify
        self._items = items

    def readData(self):
        for subfolder in os.listdir(self._folder):
            k = 0
            for file in os.listdir(os.path.join(self._folder, subfolder)):

                if k < 100:

                    x = flatten(convert2numpy(os.path.join(self._folder, subfolder,file)))
                    self._data.append(x)
                    label = [0] * self._items
                    label[int(subfolder)] = 1
                    self._labels.append(label)

                    k += 1

        #convert to numpy array
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)


        return self._data, self._labels

class NeuralNetwork:

    def __init__(self):

        #layers in the network
        self._layers = []

        #learning rate used in backprop
        self._learning_rate = 1

    def addLayer(self, layer):
        self._layers.append(layer)

    def train(self, x, y, iterations, figure = False):

        self._accurayIterations = [0] * iterations
        for iter in tqdm(range(iterations)) :
            for k in range(len(x)):
                z = self.feedforward(x[k])
                self.backprop(x[k], y[k])
                self._accurayIterations[iter] += np.sum(brier_vectorized(z, y[k]))

            self._accurayIterations[iter] /= len(x)

        if figure:
            plt.plot(self._accurayIterations, linestyle = 'dotted')
            plt.show()

    def backprop(self, x, y):

        r = list(reversed(self._layers))
        for i in range(len(r)):

            #output layer
            if i == 0:
                r[i].backprop(r[i + 1]._z, y)

            #input layer
            elif i == len(r) - 1:
                r[i].backprop(x, y, r[i - 1])

            #hidden layer
            else:
                r[i].backprop(r[i + 1]._z, y, r[i - 1])


    def feedforward(self, x):

        #going through all layers
        for i in range(len(self._layers)):

            #if first layer, pass data
            if i == 0:
                z = self._layers[i].feedforward(x)

            #else pass return from previous layers
            else:
                z = self._layers[i].feedforward(z)

        return z

    def saveModel(self, name):
        data = {}
        for i in range(len(self._layers)):
            data[i] = {}
            for k in range(len(self._layers[i]._weights)):
                for h in range(len(self._layers[i]._weights[0])):
                    data[i][str(k) + "_" + str(h)] = self._layers[i]._weights[k][h]

        json_obj = json.dumps(data)
        with open(name + ".json", "w") as outfile:
            outfile.write(json_obj)

class Layer:

    def __init__(self, network, inputs, outputs, layerType = "hidden"):

        #linear combination of values
        self._a = np.zeros(outputs)

        #output of layer after transform
        self._z = np.zeros(outputs)

        #weights
        #self._weights = np.ones((outputs, inputs))
        self._weights = np.random.random((outputs, inputs))

        #biases
        self._biases = np.ones(outputs)

        #errors
        self._deltas = np.ones(outputs)

        #type of layer: either output or hidden
        self._layerType = layerType

        #reference to parent network
        self._network = network


    def feedforward(self, x):

        #linear combinations of input from previous layer
        self._a = np.matmul(self._weights, x)

        #activation function
        self._z = sigmoid_vectorized(self._a)

        return self._z

    def backprop(self, z, y = None, nextLayer = None):

        if self._layerType == "output":

            #errors in output layer
            self._deltas = loss_vectorized(self._z, y)

            #updating with inputs from previous layer
            updates = - self._network._learning_rate * np.outer(self._deltas, z)

        else:

            #intermediate calculation
            errorsums = np.matmul(nextLayer._weights.transpose(), nextLayer._deltas)

            #errors in  layer
            self._deltas = self._z * (1 - self._z) * errorsums

            #updating with inputs from previous layer
            updates = - self._network._learning_rate * np.outer(self._deltas, z)

        #final updates on weights
        self._weights = np.add(self._weights, updates)


if __name__ == "__main__":


    '''

    nn = NeuralNetwork()

    l1 = Layer(nn, 3, 3, "hidden")
    l2 = Layer(nn, 3, 3, "output")

    data = np.array([[1,1,1],[2,2,2]])
    answer = np.array([[0,1,0],[1,0,0]])

    #data = np.array([[1,1,1]])
    #answer = np.array([[0,1]])


    #data = np.array([1,1,1])
    #answer = np.array([0,1])


    nn.addLayer(l1)
    nn.addLayer(l2)

    #print(nn.feedforward(data))

    nn.train(data, answer, 1)

    r = convert2numpy("../../data/Reduced MNIST Data/Reduced Trainging data/0/4924.jpg")

    print(r.shape)

    imgLoader = ImageDataLoader("../../data/Reduced MNIST Data/Reduced Trainging data", 10)


    l1.feedforward(data)
    l2.feedforward(l1._z)

    l2.backprop(l1._z, answer)
    l1.backprop(data, answer, l2)


    for i in range(1000):

        l1.feedforward(data)
        l2.feedforward(l1._z)

        l2.backprop(l1._z, answer)
        l1.backprop(data, answer, l2)

    print(l2._z)
    '''

    #answer from manual code: [0.53407277 0.94834531]

    print("... reading data ...")

    imgLoader = ImageDataLoader("../../data/Reduced MNIST Data/Reduced Trainging data", 10)
    data, labels = imgLoader.readData()

    print("... training model ...")

    network = NeuralNetwork()
    layer1 = Layer(network, 784, 40, "hidden")
    network.addLayer(layer1)
    layer2 = Layer(network, 40, 40, "hidden")
    network.addLayer(layer2)
    layer3 = Layer(network, 40, 10, "output")
    network.addLayer(layer3)

    network.train(data, labels, 3, True)
    network.saveModel("mini")
