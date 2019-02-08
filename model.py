# Load the data 
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import GlobalAveragePooling2D, Input, Lambda

def get_data(): 
    lines = [] 
    # with open('../data_sample/driving_log.csv') as csvfile: 
    with open('/opt/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader: 
            lines.append(line) 
        
    images = [] ## images = x in our architecture
    measurements = [] ## measurements = y in our architecture
    for line in lines[1:]: 
        source_path = line[0][18:]
        source_path = '/opt/data/IMG/' + source_path
        images.append(image)
        steering_angle = float(line[6])
        measurements.append(str(round(steering_angle))) 
        
    print('Total number of images loaded: ' + str(len(images)))
    return images, measurements

# Shuffle the data, use 70% for training, and 30% for validation 
def separate_training_and_validation_data(X_data, y_data, training_percent = 70): 
    X_data, y_data = shuffle(X_data, y_data)
    
    training_len = int(len(X_data) * training_percent / 100)
    
    X_train, y_train, X_valid, y_valid = X_data[:training_len], y_data[:training_len], X_data[training_len:], y_data[training_len:]
    print('Size of training set: ' + str(len(X_train)))
    print('Size of validation set: ' + str(len(X_valid)))
    return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid)

# Preprocess the image by normalizing it 
    
def preprocess(X_data):
    return (X_data - 128) / 128

def one_hot_encode(y_data):
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_data)
    return y_one_hot

## TODO play with those parameters, make sure that the outputs are consistant, that is what caused the issue before!!!! 
## TODO Decided what parameters are ok to modify
## TODO test the output!
def get_model():
    inception = InceptionV3(weights='imagenet', include_top=False)
    
    image_input = Input(shape=(160, 320, 3))
    
    inp = inception(image_input)
    x = GlobalAveragePooling2D()(inp)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(31, activation = 'softmax')(x)
    
    model = Model(inputs=image_input, outputs=predictions)
    model.summary()
    return model
              
    return model

def data_generator(X_train, y_train): 
    for i in range(len(X_train)):
        yield X_train[i], y_train[i]
    
def train(X_train, y_one_hot_train, X_val, y_one_hot_val, batch_size=32, epochs=5): 
    print('Training')
    model = get_model()
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(X_train.shape)
    print(y_one_hot_train.shape)
    print(X_val.shape)
    print(y_one_hot_val.shape)
    model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size), 
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 
                    validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)
    
    model.save('model.h5')

# Create the model 
# Train the model, and evaluate
# Run the 
X_data, y_data = get_data()
y_data = one_hot_encode(y_data)
X_train, y_train, X_valid, y_valid = separate_training_and_validation_data(X_data, y_data) 

X_train = preprocess(X_train)
X_valid = preprocess(X_valid) 

train(X_train, y_train, X_valid, y_valid)


