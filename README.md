# Image-detection

The Repository consists of modelling Of CNN architecture 
using keras with tensirfloe background.
We are dealing woth image classification as cat or dog from given input image.

## About Dataset
The dataset is of 10000 images in total consisting of 5000 cat and 5000 dog images. The train:test ratio is 4:1 for both of the cat and dog set. It is taken from https://www.superdatascience.com/deep-learning/ (PART 2. CONVOLUTIONAL NEURAL NETWORKS (CNN))

## Processing
The dataset is first converted to a 64 cross 64 resolution in RGB layering and passed through a Covolutional neural network where it is first convolved with 32 different feature detectors of size 3 cross 3.
And then it is MAxpooled and Flattened to get the input layer from fully coneected neural network.
Then neural network is trained to get the weights of each node.

DataGenerator is used to create more data from available data to train model better. The datagen implementation is taken from keras documentation file for image processing.
The link for the same is https://keras.io/preprocessing/image/.
The entire architecture looks like this-
![picture](https://github.com/ajinkyaambatwar/Image-detection/blob/master/Screenshot%20from%202018-04-18%2021-58-03.png)

## About deeply connected network
The deeply conneceted network consists of input layer of of nodeswhich is equal to number of node obtained after flattening and 1 output layer to determine the probabilty of the image to be a cat or dog(classification problem). The hidden layer consists of 128 nodes. 
