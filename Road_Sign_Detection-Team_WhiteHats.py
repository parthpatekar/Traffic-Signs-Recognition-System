'''
The BelgiumTS dataset is used for training and testing.
Download the training dataset: BelgiumTSC_Training, and
Download the testing dataset : BelgiumTSC_Testing; from http://btsd.ethz.ch/shareddata/
'''

# TODO: Modify the ROOT_PATH, train_data_dir and test_data_dir in the "Load training and testing datasets" section as per the respective directory paths.

# Modules used for this project are: Numpy, Scikit-image, matplotlib, tensorflow, Pandas

import os
import random
from skimage import data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

def load_data(data_dir):
    '''
    Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    '''

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load training and testing datasets.
ROOT_PATH = "/home/yash/Desktop/traffic/"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")

images, labels = load_data(train_data_dir)

# Resize images
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]

# Minimum Viable Model
labels_a = np.array(labels)
images_a = np.array(images32)

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer. 
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])

for i in range(201):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: images32})[0]
# Evaluation
# Load the test dataset.
test_images, test_labels = load_data(test_data_dir)

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]
# Run predictions against the full test set.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

directories = [d for d in os.listdir(test_data_dir) 
                   if os.path.isdir(os.path.join(test_data_dir, d))]

# Loop through the label directories and collect the data in
# two lists, labels and images.
lbl = []
img = []
for d in directories:
	label_dir = os.path.join(test_data_dir, d)
	file_names = [os.path.join(label_dir, f) 
		for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
	for f in file_names:
		f2=f.split('/')[9]

# Creating a data frame for making the CSV file
result=pd.DataFrame({'File Names':f2, 'Class':predicted})
result.to_csv('result.csv', index=False, cols=['File Names', 'Class'])

# For checking the validity of the neural network, its accuracy against the given Testing dataset can be checked.
'''
accuracy = float(match_count) / float(len(test_labels))
print("Accuracy: {:.3f}".format(accuracy))
'''

# Close the session. This will destroy the trained model.
session.close()