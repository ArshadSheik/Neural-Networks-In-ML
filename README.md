# Neural-Networks-In-ML
Implementing a Classification Model with Neural Network and Visualizing how Neural Network works.

# Neural Networks

A Neural Network is the most important concept in deep learning, which is a subset of machine learning that mimics the workings of a human brain while solving a complex problem based on deep learning. Neural networks are inspired by neurons found in the human brain.

Artificial Neural Networks are normally called Neural Networks (NN).

Neural networks are in fact multi-layer Perceptrons.

The perceptron defines the first step into multi-layered neural networks.  

&nbsp;

# How Neural Network Works ?

A Neural Network is a computational structure that connects an input layer to an output layer. This computational structure is used in training deep learning models that can easily outperform any classical machine learning algorithm.

A Neural Network consists of three types of layers:
&nbsp;
1. one input layer
2. one or more hidden layers
3. one output layer

&nbsp;
![](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2022/01/input-hidden-and-output-layer.png?resize=768%2C562&ssl=1)

&nbsp;

Let’s understand how a neural network works with an example of Image Classification. To classify images using a neural network, we will first feed the neural network with the pixel values of images. **The first layer of a neural network is the input layer that receives the data as input.**

**The second layer of a neural network is the hidden layer, responsible for all the calculations to learn from the features of the input data.** There are only three layers in a neural network, but the number of hidden layers can be increased. The more complex the problem, the more hidden layers are preferred. Typically, a neural network with 1-2 hidden layers will work in most deep learning problems, but if the data has a lot of features to learn from, we can choose 3-5 hidden layers.

**The last layer of a neural network is the output layer which classifies the data and provides the final output.** The result given by an output layer is controlled by an [**activation function**](https://github.com/ArshadSheik/Neural-Networks-In-ML/blob/master/Activation%20Functions/README.md). The activation function is placed after the hidden layer and used to calculate the weighted sum of inputs and biases, used to determine whether a neuron should be activated or not.

**So the first layer or the input layer of a neural network receives the input, the second layer or the hidden layers learn from the features of the input data, and the third layer or the output layer provides the output controlled by an activation function.**

&nbsp;

# The Neural Network Model

Input data (Yellow) are processed against a hidden layer (Blue) and modified against another hidden layer (Green) to produce the final output (Red).

![neural_network](https://www.w3schools.com/ai/img_neural_networks.jpg)

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

![](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png)

Neural networks rely on training data to learn and improve their accuracy over time. However, once these learning algorithms are fine-tuned for accuracy, they are powerful tools in computer science and artificial intelligence, allowing us to classify and cluster data at a high velocity. Tasks in speech recognition or image recognition can take minutes versus hours when compared to the manual identification by human experts. One of the most well-known neural networks is Google’s search algorithm.   

&nbsp;
&nbsp;

# Types of Neural Networks

Neural networks are classified according to their architectures. There are **7** (Major) types of Neural Networks and those are:

1. Perceptron
2. Artificial neural networks
3. Multilayer Perceptron
4. Radial networks
5. Convolutional neural networks
6. Recurrent neural networks
7. Long-term short-term memory

&nbsp;

## 1. Perceptron:

[Perceptron](https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975) is the most basic architecture of neural networks. It is also known as a single layer neural network because it contains only one input layer and one output layer. There are *no hidden layers present in the perceptron*. It works by taking inputs, then it calculates the weighted input of each node, then it uses an activation function. Perceptrons are a basic form of neural networks, **so this type of architectures is only preferred for classification-based problems.**

![perceptron](https://miro.medium.com/max/1400/1*Fyapb-JRFJ-VtnLYLLXCwg.png)


&nbsp;


## 2. Artificial Neural Networks:

An Artificial Neural Network is also known as a fast forward neural network. In this type of neural network, all perceptrons are layered such that the input layers take input and the output layers generate output. In an artificial neural network, all nodes are fully connected, which means that every perceptron in one layer is connected to every node in the next layer. **These types of neural networks are the best to use in computer vision applications.**

![](https://www.smartsheet.com/sites/default/files/IC-simplified-artificial-neural-networks-corrected.svg)


&nbsp;


## 3. Multilayer Perceptron:

Artificial Neural Networks has a shortcoming to learn with backpropagation, this is where multilayer perceptrons come in. [Multilayer perceptrons](https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141) are the types of neural networks which are bidirectional by which I mean that they forward propagation of the inputs and the backward propagation of the weights. Here all the neurons in a layer are connected to all the neurons in the next layer. The below is an example multilayer perceptron (MLP) model of a multi-classification artificial neural network (ANN):

![](https://miro.medium.com/max/1400/1*tYQrcNF2rPAETvzpqGX0PA.png)


&nbsp;


## 4. Radial Based Networks:

Multilayer perceptrons can be used in any type of deep learning application but they are slow due to their architecture, this is where radial networks come in. Radial networks are different from all types of neural networks because of their faster learning rate. The difference between a radial basis network and an artificial neural network is that [Radial Based Networks](https://towardsdatascience.com/radial-basis-functions-neural-networks-all-we-need-to-know-9a88cc053448) use a radial basis function as an activation function. **This architecture is best to use when the problem is based on classification.**

![](https://chrisjmccormick.files.wordpress.com/2013/08/architecture_simple2.png)


&nbsp;


## 5. Convolutional Neural Networks:

[Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) are one of the best types of neural networks that can be used in any computer vision task, especially in image classification. **CNN can be used for most of computer vision problems because it contains multiple layers of neurons that are used to understand the most important features of an image**. In a convolutional neural network, the first layers of neurons are used to understand lower level features and the remaining layers of neurons are used to understand high-level features.

![](https://miro.medium.com/max/1400/1*uAeANQIOQPqWZnnuH-VEyw.jpeg)


&nbsp;


## 6. Recurrent Neural Networks:

[Recurrent Neural Networks](https://towardsdatascience.com/introducing-recurrent-neural-networks-f359653d7020) are types of artificial neural networks where each neuron present inside the hidden layer receives an input with a specific delay. When we need to access the previous set of information in current iterations, it is best to use recurrent neural networks. **It can be used in very complex deep learning applications such as machine translation systems and robot control applications.**

![](https://miro.medium.com/max/1400/1*3ltsv1uzGR6UBjZ6CUs04A.jpeg)


&nbsp;


## 7. Long Short Term Memory Networks:

[Long Short Term Memory or LSTM Networks](https://towardsdatascience.com/lstm-recurrent-neural-networks-how-to-teach-a-network-to-remember-the-past-55e54c2ff22e) are used in deep learning applications where data is processed with memory gaps. The best part about LSTMs is that they can remember data for longer. So, whenever your neural network fails to remember the data, you can use LSTM networks. One of the applications where it is widely used is the prediction of time series. So we can say that **when you want to use deep learning for regression-based problems, you can use LSTM networks.**

![](https://miro.medium.com/max/1400/1*7cMfenu76BZCzdKWCfBABA.png)
