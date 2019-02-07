# Load the data 
import csv
import cv2
from sklearn.utils import shuffle

def get_data(): 
    lines = [] 
    # with open('../data_sample/driving_log.csv') as csvfile: 
    with open('data_sample/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader: 
            lines.append(line) 
        
    images = [] ## images = x in our architecture
    measurements = [] ## measurements = y in our architecture
    for line in lines: 
        source_path = line[0]
        image = cv2.imread(source_path)
        images.append(image)
        steering_angle = line[6]
        measurements.append(steering_angle) 
        
    print('Total number of images loaded: ' + str(len(images)))
    return images, measurements

# Shuffle the data, use 70% for training, and 30% for validation 
def separate_training_and_validation_data(X_data, y_data, training_percent = 70): 
    X_data, y_data = shuffle(X_data, y_data)
    
    training_len = int(len(X_data) * training_percent / 100)
    
    X_train, y_train, X_valid, y_valid = X_data[:training_len], y_data[:training_len], X_data[training_len:], y_data[training_len:]
    print('Size of training set: ' + str(len(X_train)))
    print('Size of validation set: ' + str(len(X_valid)))
    return X_train, y_train, X_valid, y_valid

# Preprocess the image
# Create the model 
# Train the model, and evaluate
# Run the 
X_data, y_data = get_data()
X_train, y_train, X_valid, y_valid = separate_training_and_validation_data(X_data, y_data) 
 
