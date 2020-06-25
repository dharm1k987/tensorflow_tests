
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# initially there are 60000 images of size 28x28
# but Conv2D expects a 4D array with the fourth parameter being the channel
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

training_images=training_images / 255.0
test_images=test_images/255.0

# convolution is used to detect patterns, and rely on image filters
# generate 64 filters of size 3x3
# filters can be specific to detect edges, curves, etc
# create a 2x2 pools, for every 4 pixels, we take the biggest pixel
# add another convolution and pool layer
model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640 -> original img is 28x28, but since we cant convulve on the edge, its shrunk by 2 px each side
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0 -> max pooling will half the image, taking the largest pixel
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928 -> convulve again
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0 -> half again
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0 -> 64 new images of 5x5 that have been fed in, so 5x5x64 = 1600
_________________________________________________________________
dense (Dense)                (None, 128)               204928
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
"""