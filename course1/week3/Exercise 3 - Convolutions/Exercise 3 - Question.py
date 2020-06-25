
import tensorflow as tf
print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') >= 0.998):
            print('\nReached 99.8% accuracy so cancelling training!')
            self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# convert the training images to a 4D array with size 60000x28x28x1
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

training_images=training_images / 255.0
test_images=test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
test_loss = model.evaluate(test_images, test_labels)