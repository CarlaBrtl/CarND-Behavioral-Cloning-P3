# Load the data 
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import GlobalAveragePooling2D, Input, Lambda, Cropping2D

def get_image_names(): 
    lines = [] 
    # with open('../data_sample/driving_log.csv') as csvfile: 
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines         
            
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

## TODO Change this
def plot_measurements(measurements):
    
    plt.bar(measurements, color="blue")
    plt.savefig("plot.jpg")

## TODO play with those parameters, make sure that the outputs are consistant, that is what caused the issue before!!!! 
## TODO Decided what parameters are ok to modify
## TODO test the output!
def get_model():
    inception = InceptionV3(weights='imagenet', include_top=False)
    for layer in inception.layers: 
        layer.trainable=False
    
    image_input = Input(shape=(160, 320, 3))
    normalized = Lambda(lambda x: (x / 255.0) - 0.5)(image_input)
    cropped = Cropping2D(cropping=((50,20), (0,0)))(normalized)
    inp = inception(normalized)
    x = GlobalAveragePooling2D()(inp)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1)(x)
    
    model = Model(inputs=image_input, outputs=predictions)
    model.summary()
    return model
              
def data_generator(X_train, y_train): 
    for i in range(len(X_train)):
        yield X_train[i], y_train[i]
    
def train(train_generator, train_samples, validation_generator, validation_samples, epochs=2): 
    print('Training')
    model = get_model()

    model.compile(loss = 'mse' , optimizer='adam')
    
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
                        validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)
    
    model.save('model.h5')
    return model 

samples = get_image_names()
# Shuffle the data, use 80% for training, and 20% for validation 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Get and compile mode 
model = get_model()
model.compile(loss='mse', optimizer='adam')

# Train the model
print("Training")
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

