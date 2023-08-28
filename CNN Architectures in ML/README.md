# CNN Architectures in Machine Learning

## Convolutional Neural Network (CNN)

Convolutional Neural Networks or CNN originate from the study of the visual cortex of the brain and have been used in image recognition since the 1980s.

CNN or convolutional neural networks are used to power image search services, self-driving cars, automatic video classification systems, etc. Plus, CNN isn’t just limited to visual perception – they are also successful in many other tasks, such as speech recognition and natural language processing.

A convolution is a mathematical operation that slides one function onto another and measures the integral of their point multiplication. It has deep connections with the Fourier transform and the Laplace transform and is heavily used in signal processing. Convolutional layers use cross-correlations, which are very similar to convolutions.

## Introduction to Convolutional Layers

![convolutionallayers](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2020/11/1-CNN.png?resize=1024%2C452&ssl=1)

The most important building block of a CNN is the convolutional layer: neurons from the first convolutional layer are not connected to every pixel of the input image, but only to the pixels of their receptive fields.

In turn, each neuron of the second convolutional layer is connected only to neurons located in a small rectangle of the first layer. This architecture allows the network to focus on small, low-level features in the first hidden layer, and then assemble them into larger, higher-level features in the next hidden layer, etc.

This hierarchical structure is common in real-world images, which is one of the reasons CNN works so well for image recognition.

## **CNN Architectures**

In Machine Learning the typical CNN architectures stack a few convolutional layers, then a pooling layer, then a few more convolutional layers, then another pooling layer, and so on. The image gets smaller and smaller as it moves through the network, but it usually gets deeper and deeper.

![cnn](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2020/11/1-cnnlayer.png?resize=1024%2C259&ssl=1)

At the top of the stack, a regular feed-forward neural network is added, made up of a few fully connected layers (+ ReLUs), and the final layer produces the prediction.

## Types of CNN Architectures

## **LeNet-5:**

LeNet-5 architecture is perhaps the most well-known CNN architecture. It was created by Yann LeCun in 1998 and has been widely used for the recognition of handwritten digits (MNIST). It is composed of the layers indicated in the table below.

| Layer  | Type            | Maps | Size    | Kernel Size | Stride | Activation |
|--------|-----------------|------|---------|-------------|--------|------------|
| Out    | Fully connected | -    | 10      | -           | -      | RBF        |
| F6     | Fully connected | -    | 84      | -           | -      | tanh       |
| C5     | Convolution     | 120  | 1 X 1   | 5 X 5       | 1      | tanh       |
| S4     | Avg Pooling     | 16   | 5 X 5   | 2 X 2       | 2      | tanh       |
| C3     | Convolution     | 16   | 10 X 10 | 5 X 5       | 1      | tanh       |
| S2     | Avg Pooling     | 6    | 14 X 14 | 2 X 2       | 2      | tanh       |
| C1     | Convolution     | 6    | 28 X 28 | 5 X 5       | 1      | tanh       |
| In     | Input           | 1    | 32 X 32 | -           | -      | tanh       |



## **AlexNet:**

The AlexNet CNN architecture won by far the ImageNet ILSVRC 2012 challenge: it achieved a top-five error rate of 17%, while the second-best only reached 26%. It was developed by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton.

It’s similar to the LeNet-5 but much larger and deeper, and it was the first to stack convolutional layers directly on top of each other, instead of stacking a grouping layer on top of each convolutional layer. The table below shows the CNN architecture of AlexNet.

<table><thead><tr><th>Layer</th><th>Type</th><th>Maps</th><th>Size</th><th>Kernel Size</th><th>Stride</th><th>Padding</th><th>Activation</th></tr></thead><tbody><tr><td>Out</td><td>Fully Connected</td><td>–</td><td>1000</td><td>–</td><td>–</td><td>–</td><td>Softmax</td></tr><tr><td>F10</td><td>Fully Connected</td><td>–</td><td>4,096</td><td>–</td><td>–</td><td>–</td><td>ReLU</td></tr><tr><td>F9</td><td>Fully Connected</td><td>–</td><td>4,096</td><td>–</td><td>–</td><td>–</td><td>ReLU</td></tr><tr><td>S8</td><td>Max pooling</td><td>256</td><td>6 X 6</td><td>3 X 3</td><td>2</td><td>valid</td><td>–</td></tr><tr><td>C7</td><td>Convolution</td><td>256</td><td>13 X 13</td><td>3 X 3</td><td>1</td><td>same</td><td>ReLU</td></tr><tr><td>C6</td><td>Convolution</td><td> 384</td><td>13 X 13</td><td>3 X 3</td><td>1</td><td>same</td><td>ReLU</td></tr><tr><td>C5</td><td>Convolution</td><td>384</td><td>13 X 13</td><td>3 X 3</td><td>1</td><td>same</td><td>ReLU</td></tr><tr><td>S4</td><td>Max pooling</td><td>256</td><td>13 X 13</td><td>3 X 3</td><td>2</td><td>valid</td><td>–</td></tr><tr><td>C3</td><td>Convolution</td><td>256</td><td>27 X 27</td><td>5 X 5</td><td>1</td><td>same</td><td>ReLU</td></tr><tr><td>S2</td><td>Max pooling</td><td>96</td><td>27 X 27</td><td>3 X 3</td><td>2</td><td>valid</td><td>–</td></tr><tr><td>C1</td><td>Convolution</td><td>96</td><td>55 X 55</td><td>11 X 11</td><td>4</td><td>valid</td><td>ReLU</td></tr><tr><td>In</td><td>Input</td><td>3 (RGB)</td><td>227 X 227</td><td>–</td><td>–</td><td>–</td><td>–</td></tr></tbody></table>



## **GoogLeNet:**

The GoogLeNet architecture was developed by Christian Szegedy et al. from Google Research, and he won the 2014 ILSVRC challenge by pushing the top five error rate below 7%. This excellent performance came in large part from the fact that the network was much deeper than previous CNNs.

GoogleNet was made possible by subnets called starter modules, which allow GoogLeNet to use parameters much more efficiently than previous architectures: GoogLeNet actually has 10 times fewer parameters than AlexNet (around 6 million instead of 60 million).

The image below represents the CNN architecture of GoogleNet.

![googlenet](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2020/11/1-GoogleNet.png?w=1009&ssl=1)



## **VGGNet:**

The finalist for the 2014 ILSVRC Challenge was VGGNet, developed by Karen Simonyan and Andrew Zisserman of the Visual Geometry Group (VGG) research laboratory at the University of Oxford.

It had a very simple and classic architecture, with 2 or 3 convolutional layers and a pooling layer, then again 2 or 3 convolutional layers and a pooling layer, and so on (reaching a total of only 16 or 19 convolutional layers). , according to VGG), plus a final dense network with 2 hidden layers and the exit layer. He only used 3 × 3 filters, but lots of filters.



## **ResNet:**

Kaiming He et al. won the 2015 ILSVRC challenge using a residual network (or ResNet), which produced a staggering error rate in the top five under 3.6%. The winning variant used an extremely deep CNN consisting of 152 layers (the other variants had 34, 50 and 101 layers).

This confirmed the general trend: models are getting deeper and deeper, with fewer and fewer parameters. The key to being able to form such a deep network is to use jump connections (also known as shortcut connections): the signal feeding one layer is also added to the output of a layer further up the stack.

The image below represents the CNN architecture of ResNet.

![resnet](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2020/11/1-resnet.png?resize=768%2C443&ssl=1)<br>



## **SENet:**

The winning architecture of the ILSVRC 2017 challenge was the Squeeze-and-Excitation (SENet) network. This architecture extends existing architectures such as boot networks and reboots and improves their performance.

This allowed SENet to win the competition with an astonishing 2.25% error rate in the top five! The extended versions of Bootnets and ResNets are called SE-Inception and SE-ResNet, respectively. The boost comes from the fact that a SENet adds a small neural network, called an SE block, to each unit of the original architecture.
