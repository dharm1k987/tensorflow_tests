from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
import os

# 300x300 image, and RGB (3 bytes)
# in binary classifications, sigmoid is best used
# it is a scalar between 0 and 1 representing which class it favours
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(300, 300, 3)),
    keras.layers.Conv2D(16, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 298, 298, 16)      448 -> 298x298 image after 3x3 filter
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 149, 149, 16)      0 -> pool it
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 147, 147, 32)      4640 -> filter again
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 73, 73, 32)        0 -> pool again
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 71, 71, 64)        18496 -> filter again
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 35, 35, 64)        0 -> pool again
_________________________________________________________________
flatten (Flatten)            (None, 78400)             0 -> 64 convolutions that are 35x35 gives us 64x35x35 = 78400
_________________________________________________________________
dense (Dense)                (None, 512)               40141312
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513
=================================================================
"""

# compile model
# binary loss function
# optimizer in which we can control the learning rate
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

# data preprocessing
# data generators will read pictures from our source folders
# convert them to float32 tensors, and feed them with their labels to the network
# we have two generators, one for training, and one for validation images

# first we normalize the data
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        os.path.join('./horse-or-human'),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128, # default
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 32 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        os.path.join('./validation-horse-or-human'),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


# fit model
# train_generator is used to stream the images from the training directory
# steps_per_epoch = total number of images in training / batch size = 1024 / 128 = 8
# total epochs to run for
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data= validation_generator,
    validation_steps=8
)
