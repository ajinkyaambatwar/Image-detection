# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
#PArt1-Building CNN
from keras.models import Sequential      #initiates CNN
from keras.layers import Conv2D,Convolution2D   #Does 2d convolution
from keras.layers import MaxPooling2D    #Does MAxPooling
from keras.layers import Flatten         #Does Flattening
from keras.layers import Dense           #Adds fully connected network

#INitializing CNN
classifier=Sequential()

#Step 1-Convolution
'''Creating 32 feature detectors of six=ze 3*3'''
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3)))
'''MAx pooling with pool size of 2*2'''
classifier.add(MaxPooling2D(pool_size=(2,2)))
'''Flattening'''
classifier.add(Flatten())
'''Full connection'''
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

'''Compiling'''
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

'''Fitting the CNN to the data'''
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(                                 #Apply some transformations to get more data from availabel data
        rescale=1./255,                                                     #Rescale pixel values to 0 to 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
            'Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/training_set',
            target_size=(64, 64),           #equal to input shape
            batch_size=32,
            class_mode='binary')
    
test_set = test_datagen.flow_from_directory(
            'Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)