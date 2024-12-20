#Python code: mnist_full_bayes.py Laura Nyrhilä
import numpy as np
import tensorflow as tf
import sys
from scipy.stats import multivariate_normal



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
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = np.add(x_train, np.random.normal(loc=0.0, scale=10.0, size=x_train.shape))  # adding noise

#normalize data with min-max normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

#identify classes and features
classes = np.unique(y_train)
classes_len = len(classes)
N_features = x_train.shape[1]

#initialize vectors (mean and variance)
mean_vectors = np.zeros((classes_len, N_features), dtype=np.float32)
cov_matrices = np.zeros((classes_len, N_features, N_features), dtype=np.float32)

for c in classes:
    x_class = x_train[y_train == c]  # Selecting all images belonging to class c
    mean_vectors[c, :] = np.mean(x_class, axis=0)
    cov_matrices[c, :, :] = np.cov(x_class, rowvar=False) + np.eye(N_features) * 1e-6  # Added small value to ensure positive definiteness

log_likelihoods = np.zeros((x_test.shape[0], classes_len), dtype=np.float32)
for c in range(classes_len):
    log_likelihoods[:, c] = multivariate_normal.logpdf(x_test, mean=mean_vectors[c, :], cov=cov_matrices[c, :, :])

y_pred = np.argmax(log_likelihoods, axis=1)
# Calculate and print accuracy
accuracy = acc(y_pred, y_test)
print(f'Classification accuracy is {accuracy:.2f}%')