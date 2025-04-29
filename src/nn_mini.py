
import os
import json
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

#Making the output human readable
np.set_printoptions(precision=2)

###########################################
# Different vectorizations and functions
###########################################

#loss function where output in a sigmoid (y_pred is return from sigmoid function)
def loss_derivative(y_pred, y):
    return (y_pred - y)

#sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + 2.71 ** (-x))

#brier score
def brier(y_pred, y):
    return (y_pred - y)**2

#softmax
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def cross_entropy(ypred, y):
    return np.sum(y * np.log(ypred + 1e-9))

def maxList(x):
    maxVal = -1
    index = None
    k = 0
    for v in x:
        if v > maxVal:
            maxVal = v
            index = k
        k+=1
    return index


def mat2markdown(m):

    s = "| Actual | Predicted |" + "\n"

    for k in range(len(m)):
        s += "| --- "
    s+= "| \n"

    for k in range(len(m)):
        s +="|" + str(k)
    s+= "| \n"

    for k in range(len(m)):
        s += "|" + str(k) + "|"
        for h in range(len(m)):
            s += str(m[k][h]) + "|"
        s+= "\n"

    return s




#vectorizations
loss_vectorized = np.vectorize(loss_derivative)
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

    def __init__(self, folder, items, scale = 1):

        #folder where subfolder for each category is placed
        self._folder = folder

        #data to be returned
        self._data = []

        #labels
        self._labels = []

        #number of categories to classify
        self._items = items

        self._scale = scale

    def readData(self):
        for subfolder in os.listdir(self._folder):
            k = 0
            for file in os.listdir(os.path.join(self._folder, subfolder)):

                if k < 40:

                    x = flatten(convert2numpy(os.path.join(self._folder, subfolder,file))) / self._scale
                    self._data.append(x)
                    label = [0] * self._items
                    label[int(subfolder)] = 1
                    self._labels.append(label)

                    #k += 1

        #convert to numpy array
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)

        #shuffle
        data = list(zip(self._data, self._labels))
        random.shuffle(data)

        self._data, self._labels = zip(*data)

        return self._data, self._labels

class NeuralNetwork:

    def __init__(self):

        #layers in the network
        self._layers = []

        #learning rate used in backprop
        self._learning_rate = .01

    def addLayer(self, layer):
        self._layers.append(layer)

    def train(self, x, y, iterations, figure = False):

        #Overall value of loss function
        self._accurayIterations = [0] * iterations

        for iter in tqdm(range(iterations)) :
            for k in range(len(x)):

                #Feeding forward values in the network (for one single item)
                z = self.feedforward(x[k])

                #backpropagation through network
                self.backprop(x[k], y[k])

                #recording loss
                self._accurayIterations[iter] += np.sum(cross_entropy(z, y[k]))



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
            data[i] = {'type': self._layers[i]._layerType, 'shape': self._layers[i]._weights.shape, "weights": {}}

            for k in range(len(self._layers[i]._weights)):
                for h in range(len(self._layers[i]._weights[0])):
                    data[i]["weights"][str(k) + "_" + str(h)] = self._layers[i]._weights[k][h]

        json_obj = json.dumps(data)
        with open(name + ".json", "w") as outfile:
            outfile.write(json_obj)

    def readModel(self, model_file):
        with open(model_file) as json_file:
            data = json.load(json_file)

        sortedLayerList = [None] * len(data)
        for layerNumber, layerData in data.iteritems():

            input = layerData['shape'][1]
            output = layerData['shape'][0]

            newLayer = Layer(self, input, output, layerData['type'])
            newLayer._weights = np.zeros((output,input)) #reset weights

            for i in range(output):
                for j in range(input):
                    newLayer._weights[i][j] = layerData['weights'][str(i) + "_" + str(j)]

            sortedLayerList[int(layerNumber)] = newLayer

        for layer in sortedLayerList:
            self.addLayer(layer)

    def predict(self, x):
        return np.round(self.feedforward(x),2)

    def evaluate(self, x, y, report = False):

        #Confusion matrix
        classNum = len(y[0]) #number of classes
        self._confusionMatrix = np.zeros((classNum, classNum))

        for k in range(len(x)):

            #Feeding forward values in the network (for one single item)
            z = self.feedforward(x[k])

            self._confusionMatrix[y[k].tolist().index(1)][maxList(z.tolist())] += 1

        #Accuracy
        accuracy = np.trace(self._confusionMatrix) / sum(sum(self._confusionMatrix))

        #Precison
        precision = np.diagonal(self._confusionMatrix) / np.sum(self._confusionMatrix, axis = 0)

        #Recall
        recall = np.diagonal(self._confusionMatrix) / np.sum(self._confusionMatrix, axis = 1)

        if report:
            print("The model's score is as follows: \n")
            print("Global accuracy is:", accuracy)

            print("Confusion matrix")
            print(pd.DataFrame(self._confusionMatrix))
            print("\n")

            print("Precison")
            print(pd.DataFrame(precision))
            print("\n")

            print("Recall")
            print(pd.DataFrame(recall))
            print("\n")



        return pd.DataFrame(self._confusionMatrix), accuracy, pd.DataFrame(precision), pd.DataFrame(recall)










class Layer:

    def __init__(self, network, inputs, outputs, layerType = "hidden"):

        #linear combination of values
        self._a = np.zeros(outputs)

        #output of layer after transform
        self._z = np.zeros(outputs)

        #weights
        #self._weights = np.ones((outputs, inputs))
        self._weights = (-1 + 2 * np.random.random((outputs, inputs)))

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
        if self._layerType == "hidden":
            self._z = sigmoid_vectorized(self._a)
        else:
            self._z = softmax(self._a)

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

    #------------------------------------
    # Data import
    #------------------------------------

    print("... reading data ...")

    #Creating an image loader from MNIST dataset
    imgLoader = ImageDataLoader("../../data/Reduced MNIST Data/Reduced Trainging data", 10, 255)

    #Fetching data
    data, labels = imgLoader.readData()

    #------------------------------------
    # Model definition
    #------------------------------------

    #The model reads flattened 28x28 pixels in a 784-length vector
    #There is two hidden layers with 50 nodes and one output layer with ten nodes (0-9)

    #Network object
    network = NeuralNetwork()

    #Defining and adding layer
    layer1 = Layer(network, 784, 50, "hidden")
    network.addLayer(layer1)

    #Defining and adding layer
    layer2 = Layer(network, 50, 50, "hidden")
    network.addLayer(layer2)

    #Defining and adding layer
    layer3 = Layer(network, 50, 10, "output")
    network.addLayer(layer3)


    #---------------------------------------
    # Training, evaluating and saving model
    #---------------------------------------

    print("... training model ...")

    #The model is trained on data and labels from the loader
    #There is no batch training, and there are five epochs.
    #The last argument "True" indicates that a plot of the loss function is given at the end.
    network.train(data, labels, 10)

    #The model is saved in a file called "mini" (.json file)
    network.saveModel("mini")


    network.evaluate(data, labels, True)

    cf, ac, pr, rec = network.evaluate(data, labels)

    #print(mat2markdown(cf))


    #------------------------------------
    # Reading in trained model
    #------------------------------------

    network2 = NeuralNetwork()
    network2.readModel("mini.json")

    #Comparing outputs from two networks
    print("1",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/1/936.jpg"))/ 255.0)* 100)
    print("1",network2.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/1/936.jpg"))/ 255.0)* 100)

    print("9",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/9/810.jpg"))/ 255.0) * 100)
    print("9",network2.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/9/810.jpg"))/ 255.0) * 100)

    #Testing on trained network
    print("7",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/7/829.jpg"))/ 255.0) * 100)
    print("5",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/5/787.jpg"))/ 255.0) * 100)
    print("7",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/7/830.jpg"))/ 255.0) * 100)
    print("2",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/2/833.jpg"))/ 255.0) * 100)
    print("6",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/6/759.jpg"))/ 255.0) * 100)
    print("6",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/6/760.jpg"))/ 255.0) * 100)
    print("8",network.predict(flatten(convert2numpy("../../data/Reduced MNIST Data/Reduced Testing data/8/775.jpg"))/ 255.0) * 100)
