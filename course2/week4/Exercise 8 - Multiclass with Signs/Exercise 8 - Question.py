import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# since the data is given to us in csv format, we have to first parse it
# each row contains 785 separate elements
# the first element is the label (letter represented in sign language)
# the other 784 are the pixels that represent the image
def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=",")
        first_line = True
        images = []
        labels = []

        for row in csv_reader:
            # ignore the first row as its just headers
            if first_line:
                first_line = False
            else:
                labels.append(row[0])
                # convert the 784 numbers into a 28x28 array and add to images
                images.append(np.array_split(row[1:785], 28))
    return np.array(images).astype('float'), np.array(labels).astype('float')

# grab the data
path_sign_mnist_train = './sign_mnist_train.csv'
path_sign_mnist_test = './sign_mnist_test.csv'

training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# now, we will need to feed this information in the ImageDataGenerator
# but the ImageDataGenerator expects 4D data, with the last dim representing the channel
# so we must expand our information from 3D to 4D
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# now we define our model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # we use softmax because there is more than one output
    tf.keras.layers.Dense(26, activation='softmax')
])

# compile our model
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# create our generators, but in a different way
# since we are not getting information from a directory
train_generator = train_datagen.flow(
    training_images,
    training_labels,
    batch_size=32
)

validation_generator = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=32
)

# fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(training_images)/32,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=len(testing_images)/32
)
