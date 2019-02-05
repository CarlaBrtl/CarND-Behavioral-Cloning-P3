# Load the data 
import csv
import cv2
import matplotlib.pyplot as plt

lines = []
# with open('../data_sample/driving_log.csv') as csvfile: 
with open('data_sample/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader: 
        lines.append(line) 
        
images = []
measurements = []
for line in lines: 
    source_path = line[0]
    image = cv2.imread(source_path)
    
    plt.imshow(image) 
    plt.show()
# Shuffle the data, use 70% for training, and 30% for validation 
# Preprocess the image
# Create the model 
# Train the model, and evaluate
# Run the 