

# Installation

Run pip install -r requirements.txt.
Preferably in a [virtual environment](https://docs.python.org/3/library/venv.html)

# What it does

The network is meant as a learning tool and is implemented using no other computational tools than numpy. However, it has some limitations (in its current state):

* No batch learning (updates are done after each item)
* Sigmoid activation at hidden neurons only
* Softmax at output layer only
* Cross entropy loss only
* The model is trained using backpropagation
* There are no biases

The program structure should be easily amenable to remedy these limitations if time allows in the future ...
Separating the classes into different files should also be done ...

# Validation

The model is validated by running ten epochs on the MNIST data set with a learning rate of 0.01. I trained it on MacBook Air from 2017 on an Intel 1.8 GHz I5 processor and it took about a minute to complete.

The model's global accuracy score is

The confusion matrix, recall and precision vectors are given below:


```python
Confusion matrix (raw counts)
0      1      2      3      4      5      6      7      8      9
0  960.0    0.0    3.0    8.0    3.0   10.0    6.0    0.0    7.0    3.0
1    0.0  986.0    7.0    0.0    0.0    2.0    2.0    1.0    1.0    1.0
2    4.0   15.0  909.0   22.0    8.0    2.0   16.0    6.0   15.0    3.0
3    3.0    3.0    9.0  942.0    1.0   11.0    1.0    4.0   19.0    7.0
4    0.0    7.0    0.0    0.0  949.0    0.0    9.0    2.0    2.0   31.0
5   14.0   11.0    1.0   63.0    7.0  822.0   24.0    4.0   46.0    8.0
6    2.0    5.0    5.0    0.0    6.0    6.0  975.0    0.0    0.0    1.0
7    0.0    7.0    6.0    8.0    9.0    0.0    0.0  935.0    6.0   29.0
8    1.0   26.0   15.0   49.0    4.0   17.0    6.0    3.0  866.0   13.0
9    3.0    2.0    4.0   13.0   58.0    6.0    2.0   36.0   13.0  863.0

```

```python
Precison (fractions)
0  0.972644
1  0.928437
2  0.947862
3  0.852489
4  0.908134
5  0.938356
6  0.936599
7  0.943491
8  0.888205
9  0.899896
```

```python
Recall (fractions)
0  0.960
1  0.986
2  0.909
3  0.942
4  0.949
5  0.822
6  0.975
7  0.935
8  0.866
9  0.863
```

# Basic use

Below is an example of how to train and use the network for predictions, including saving and loading models.
The data is just a reduced version of the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist).

```python

  #------------------------------------
  # Data import
  #------------------------------------

  print("... reading data ...")

  #Creating an image loader from MNIST dataset. 10 classes, scaling factor of 255.
  imgLoader = ImageDataLoader("../../data/Reduced MNIST Data/Reduced Trainging data", 10, 255)

  #Fetching data
  data, labels = imgLoader.readData()

  #------------------------------------
  # Model definition
  #------------------------------------

  #The model reads a flattened 28x28 pixels in a 784-length vector
  #There are two hidden layers with 50 nodes and one output layer with ten nodes (0-9)

  #Network object
  network = NeuralNetwork()

  #Defining and adding layer (784 inputs, 50 output neurons)
  layer1 = Layer(network, 784, 50, "hidden")
  network.addLayer(layer1)

  #Defining and adding layer (50 input neurons, 50 output neurons)
  layer2 = Layer(network, 50, 50, "hidden")
  network.addLayer(layer2)

  #Defining and adding layer (50 input neurons, 10 output classes)
  layer3 = Layer(network, 50, 10, "output")
  network.addLayer(layer3)


  #------------------------------------
  # Training and saving model
  #------------------------------------

  print("... training model ...")

  #The model is trained on data and labels from the loader
  #There is no batch training, and there are five epochs
  #The last argument "True" indicates that a plot of the loss function is given at the end.
  network.train(data, labels, 5, True)

  #The model is saved in a file called "mini" (.json file)
  network.saveModel("mini")


  #------------------------------------
  # Loading a trained model
  #------------------------------------

  #Define a new network object
  network2 = NeuralNetwork()

  #Read the corresponding model file
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


```
