

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

| Actual | Predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|0|1|2|3|4|5|6|7|8|9|
|0|4.0|2.0|0.0|5.0|0.0|20.0|3.0|1.0|2.0|3.0|
|1|3.0|4.0|0.0|0.0|3.0|17.0|2.0|0.0|6.0|5.0|
|2|0.0|3.0|0.0|0.0|5.0|27.0|0.0|2.0|2.0|1.0|
|3|1.0|2.0|0.0|2.0|0.0|30.0|0.0|2.0|1.0|2.0|
|4|1.0|5.0|2.0|1.0|6.0|13.0|0.0|0.0|4.0|8.0|
|5|3.0|1.0|3.0|0.0|0.0|25.0|1.0|1.0|1.0|5.0|
|6|2.0|6.0|3.0|2.0|2.0|12.0|1.0|5.0|2.0|5.0|
|7|3.0|3.0|0.0|1.0|6.0|13.0|0.0|1.0|4.0|9.0|
|8|1.0|7.0|1.0|1.0|0.0|19.0|0.0|1.0|8.0|2.0|
|9|2.0|4.0|2.0|1.0|7.0|11.0|0.0|4.0|2.0|7.0|
