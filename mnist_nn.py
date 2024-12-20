#Python code: mnist nn.py Laura Nyrhilä

import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import sys

# Evaluation function
def acc(pred, gt):
    # Ensure both pred and gt are numpy arrays
    pred = np.array(pred)
    gt = np.array(gt)
    
    # Check if both arrays have the same length
    if pred.shape != gt.shape:
        raise ValueError("Predicted and ground truth labels must have the same shape.")
    
    # Count number of correct predictions
    N_true = np.sum(pred == gt)
    
    # Calculate accuracy percent
    accuracy = N_true / len(gt) * 100
    
    return accuracy

# Get command line argument
if len(sys.argv) != 2:
    raise ValueError("Please provide exactly one argument: 'original' or 'fashion'.")

user_choice = sys.argv[1].strip().lower()

if user_choice == 'original':
    mnist = tf.keras.datasets.mnist
elif user_choice == 'fashion':
    mnist = tf.keras.datasets.fashion_mnist
else:
    raise ValueError("Please choose 'original' or 'fashion'.")

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data into 1D vectors of size 28*28 = 784
x_train_flat = x_train.reshape(x_train.shape[0], -1)  # (60000, 784)
x_test_flat = x_test.reshape(x_test.shape[0], -1)     # (10000, 784)

#normalize data
x_train_flat = x_train_flat / 255.0
x_test_flat = x_test_flat / 255.0

# Initialize the 1-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier using the training data
knn.fit(x_train_flat, y_train)

# Predict the labels for the test data
y_pred = knn.predict(x_test_flat)

# Calculate and print accuracy
accuracy = acc(y_pred, y_test)
print(f'Classification accuracy is {accuracy:.2f}%')

