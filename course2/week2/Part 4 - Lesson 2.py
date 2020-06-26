import tensorflow as tf
import os

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # flatten the results to put in dense neural network
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    # 3 outputs, so we use softmax instead of sigmoid
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

# could also have used adam here
# notice we use categorical_crossentropy instead of binary_crossentropy because there are
# more than 2 choices
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
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
# notice that the class_mode is categorical instead of binary
train_generator = train_datagen.flow_from_directory(
    os.path.join('./rps'),
    batch_size=126,
    class_mode='categorical',
    target_size=(150, 150)
)

# notice that the class_mode is categorical instead of binary
validation_generator = test_datagen.flow_from_directory(
    os.path.join('./rps-test-set'),
    batch_size=126,
    class_mode='binary',
    target_size=(150, 150)
)

# run the model

history = model.fit(train_generator,
                validation_data=validation_generator,
                steps_per_epoch=20,
                epochs=25,
                validation_steps=3,
                verbose=1
)