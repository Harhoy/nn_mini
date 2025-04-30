

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

The model's global accuracy score is 84.7 % on the test dataset.

The confusion matrix, recall and precision vectors are given below.


```python
Confusion matrix (raw counts)
       0      1      2      3      4      5      6      7      8      9
0  186.0    0.0    0.0    1.0    0.0   12.0    1.0    0.0    0.0    0.0
1    0.0  187.0    3.0    1.0    0.0    0.0    0.0    2.0    7.0    0.0
2    5.0    2.0  159.0    5.0    2.0    1.0    8.0    5.0   12.0    1.0
3    0.0    0.0    5.0  166.0    0.0   10.0    0.0    2.0   17.0    0.0
4    1.0    1.0    0.0    1.0  165.0    4.0    8.0    4.0    2.0   14.0
5    4.0    3.0    1.0   14.0    4.0  153.0   11.0    1.0    8.0    1.0
6    4.0    0.0    1.0    1.0    0.0    3.0  191.0    0.0    0.0    0.0
7    0.0    1.0    6.0    0.0    1.0    1.0    0.0  187.0    0.0    4.0
8    4.0    5.0    3.0    4.0    6.0   18.0    3.0    3.0  146.0    8.0
9    1.0    0.0    0.0    0.0   16.0    4.0    1.0   18.0    6.0  154.0

```

```python
Precison (fractions)
0  0.907317
1  0.939698
2  0.893258
3  0.860104
4  0.850515
5  0.742718
6  0.856502
7  0.842342
8  0.737374
9  0.846154
```

```python
Recall (fractions)
0  0.930
1  0.935
2  0.795
3  0.830
4  0.825
5  0.765
6  0.955
7  0.935
8  0.730
9  0.770
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

  #Creating an image loader from MNIST dataset (test data)
  imgLoaderTest = ImageDataLoader("../../data/Reduced MNIST Data/Reduced Testing data", 10, 255)

  #Fetching data
  data_test, labels_test = imgLoaderTest.readData()

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
  network.train(data_test, labels_test, 5, True)

  #The model is saved in a file called "mini" (.npy)
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
