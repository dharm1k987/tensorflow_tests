import tensorflow as tf
import numpy as np
from tensorflow import keras


def gen_vals(x):
    return x*2 - 2

# units is the number of neurons in the layer
# input_shape is how many inputs go into the first layer
# 1D array, so no need to flatten
model = tf.keras.Sequential([
    keras.layers.InputLayer(input_shape=(1,)),
    # keras.layers.Dense(3, activation='relu'),
    # keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(units=1)
])

# loss fn measures guessed answer against true answers
# optimizer is used to make guesses
model.compile(optimizer='sgd', loss='mae', metrics=['accuracy'])

xs = []
ys = []
for x in range(-10, 25):
    xs.append(x)
    ys.append(gen_vals(x))

xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)

# try to fit within 50 iterations
model.fit(xs, ys, epochs=500)

print(model.weights)

print(model.predict([24]))