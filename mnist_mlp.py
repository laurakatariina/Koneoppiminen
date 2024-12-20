#laura nyrhilä mnist_mlp
import numpy as np
import sys
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

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

# Reshape the data, flatten the images (28x28 = 784 features)
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# Scaling the dataset using MinMaxScaler (you can switch to StandardScaler if needed)
scaler = MinMaxScaler()  # or StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# One-hot encoding the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


epochs = 30
learning_rate = 0.01

# Build the network with multiple hidden layers 
model = tf.keras.models.Sequential()
# Input layer and hidden layers with sigmoid activation
model.add(tf.keras.layers.Dense(256, input_dim=784, activation='sigmoid'))
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
#output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model with Adam optimizer, cross-entropy loss, and accuracy metric
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
tr_history = model.fit(x_train, y_train, epochs=epochs)

# Evaluate the model on training data
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
print(f'Training Accuracy: {train_acc * 100:.2f}%')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Plot the training loss curve
plt.plot(tr_history.history['loss'], label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



