#Python code: mnist_naive_bayes.py Laura Nyrhilä

import numpy as np
import tensorflow as tf
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

#function for calculating log likelihood of a sample in gaussian distribution (equation (4)) 
def log_likelihood(x, mean, var):
        log_likelihoods = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_likelihoods

#function classifying test samples based on log_likelihood
def classify():
        predictions = []
        for x in x_test:
            log_likelihoods = []
            for i in range(len(classes)):
                likelihood = log_likelihood(x, mean_vectors[i, :], var_vectors[i, :])
                log_likelihoods.append(likelihood)
            predictions.append(np.argmax(log_likelihoods))  # Select class with highest log likelihood
        return np.array(predictions)

# Get command line argument
if len(sys.argv) != 2:
    raise ValueError("Please provide only one argument: 'original' or 'fashion'.")

user_choice = sys.argv[1].strip().lower()

if user_choice == 'original':
    mnist = tf.keras.datasets.mnist
elif user_choice == 'fashion':
    mnist = tf.keras.datasets.fashion_mnist
else:
    raise ValueError("Choose 'original' or 'fashion'.")

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_test = np.reshape(x_test, (10000, 784))
x_train = np.reshape(x_train, (60000, 784))

x_train = np.add(x_train, np.random.normal(loc=0.0, scale=10.0, size=x_train.shape))  # adding noise

#normalize data with min-max normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

#identify classes and features
classes = np.unique(y_train)
N_features = x_train.shape[1]

#initialize vectors (mean and variance)
mean_vectors = np.zeros((len(classes), N_features))
var_vectors = np.zeros((len(classes), N_features))

for k in classes:
        x_class = x_train[y_train == k]  # Selecting all images belonging to class k
        mean_vectors[k, :] = x_class.mean(axis=0)
        var_vectors[k, :] = x_class.var(axis=0)


y_pred = classify()
# Calculate and print accuracy
accuracy = acc(y_pred, y_test)
print(f'Classification accuracy is {accuracy:.2f}%')


