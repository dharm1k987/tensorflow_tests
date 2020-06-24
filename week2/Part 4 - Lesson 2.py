import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('loss') < 0.4):
            print('\nLoss is low so cancelling training')
            self.model.stop_training = True

callbacks = myCallback()

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# images are 28x28
# hidden layer has 128 neurons in it
# 10 outside layers representing numbers 0-9

# sequential means a sequence of layers
# flatten means turn into 1D set
# dense adds a layer of neurons
# relu means if X>0 return X else 0
# softmax takes the set of values in the last layer, and picks the largest one
    #  [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05] -> [0,0,0,0,1,0,0,0,0]
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
list(map(lambda x: print('{0:.10f}'.format(x)), classifications[0]))

print('The real value is {}'.format(test_labels[0]))
