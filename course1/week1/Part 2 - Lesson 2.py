import tensorflow as tf
import numpy as np
from tensorflow import keras

# units is the number of neurons in the layer
# input_shape is how many inputs go into the first layer
# 1D array, so no need to flatten
model = tf.keras.Sequential([
    keras.layers.InputLayer(input_shape=(1,)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1)
])

# loss fn measures guessed answer against true answers
# optimizer is used to make guesses
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# try to fit within 50 iterations
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))