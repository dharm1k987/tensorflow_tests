import tensorflow as tf
from tensorflow import keras


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') >= 0.99):
            print('\nReached 99% accuracy so cancelling training!')
            self.model.stop_training = True

callbacks = myCallback()


mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

classifications = model.predict(x_test)
list(map(lambda x: print('{0:.10f}'.format(x)), classifications[0]))

print('The real value is {}'.format(y_test[0]))