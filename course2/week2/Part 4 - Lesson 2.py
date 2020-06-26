import tensorflow as tf
import os

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # flatten the results to put in dense neural network
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    # only 1 output neuron, it will contain value 0-1 where 0 = cat, 1 = dog
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

# could also have used adam here
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics= ['accuracy']
)

# preprocess the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# image augmenting helps with not experiencing overfitting
# which in short means that it can do very well with images it has seen before
# but not so well with images it hasn't
# if you do, this means you are overfitting
# image augmenting can help reduce overfitting
# however, image augmenting will not always help
# for example, if we want to classify humans vs horses
# if we use image augmentation, then we might rotate the horse, make it upside down
# etc, but in the validation set, this will not be the case
# since horses are almost always upright
# in this case we will be overfitting
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40, # image will rotate a random amt between 0 - 40 degrees
    width_shift_range=0.2, # offset image by 20% left/right
    height_shift_range=0.2, # offset image by 20% up/down,
    shear_range=0.2, # skew image 20%
    zoom_range=0.2, # zoom any amount up till 20%
    horizontal_flip=True,
    fill_mode='nearest' # fills in any pixels that might be lost by the operation    
)

test_datagen = ImageDataGenerator(rescale=1/255)

# training images
train_generator = train_datagen.flow_from_directory(
    os.path.join('../week1/cats_and_dogs_filtered/train'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join('../week1/cats_and_dogs_filtered/validation'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

# run the model

history = model.fit(train_generator,
                validation_data=validation_generator,
                steps_per_epoch=100, # 2000 training / 20 = 100
                epochs=100,
                validation_steps=50, # 1000 test / 20 = 50 
                verbose=2
)