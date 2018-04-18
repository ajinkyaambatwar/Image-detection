# Image-detection

The Repository consists of modelling Of CNN architecture 
using keras with tensirfloe background.
We are dealing woth image classification as cat or dog from given input image.

The dataset is first converted to a 64 cross 64 resolution in RGB layering and passed through a Covolutional neural network where it is first convolved with 32 different feature detectors of size 3 cross 3.
And then it is MAxpooled and Flattened to get the input layer from fully coneected neural network.
Then neural network is trained to get the weights of each node.

DataGenerator is used to create more data from available data to train model better. The datagen implementation is taken from keras documentation file for image processing.
The link for the same is https://keras.io/preprocessing/image/.
The entire architecture looks like this-


