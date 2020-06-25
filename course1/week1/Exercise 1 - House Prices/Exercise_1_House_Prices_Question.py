# https://github.com/dharm1k987/dlaicourse/blob/master/Exercises/Exercise%201%20-%20House%20Prices/Exercise_1_House_Prices_Question.ipynb


import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([
    keras.layers.InputLayer(input_shape=(1,)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [1, 2, 3, 4, 5, 6]
ys = [1, 1.5, 2, 2.5, 3, 3.5]

model.fit(xs, ys, epochs=500)

print(model.predict([20.0]))