import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


# ### Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(f"Training Data Shape: {train_images.shape}")
print(f"Number of Training Labels: {len(train_labels)}")

print(f"Te Data Shape: {test_images.shape}")
print(f"Number of Training Labels: {len(train_labels)}")
test_images.shape


# ### Plot Image Inputs Figure

def plot_image(n):
    for i in range(n):
        plt.figure()
        plt.imshow(train_images[i])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        
plot_image(5)


# ### Scale Image Color Values (0-1)
train_images = train_images / 255.0

test_images = test_images / 255.0

plot_image(5)


# ### Plot First 25 Images & Validate Lables

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# # Model Build

# ### Build layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), ## Flattens image from (28,28) to 784 flat pixels
    keras.layers.Dense(128, activation='relu'), ## Central Layer
    keras.layers.Dense(10) ## Output Layer - 10 matching the number of possible inputs
])


# ### Complile Model Function
# Add additional setup arguements
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


## Loss function - This measures how accurate the model is during training. 
## You want to minimize this function to "steer" the model in the right direction.

## Optimizer —This is how the model is updated based on the data it sees and its loss function.

## Metrics —Used to monitor the training and testing steps. 
## The following example uses accuracy, the fraction of the images that are correctly classified.


# ### Model Training
model.fit(train_images, train_labels, epochs=10)

