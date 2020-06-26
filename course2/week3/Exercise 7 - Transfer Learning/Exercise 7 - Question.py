import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# get the weights for the NN
local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# don't use the dense layers, but provide the input shape
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False, 
    weights=None
)

# load in the weights
pre_trained_model.load_weights(local_weights_file)

# all of the layers should be non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# we will end at the same layer as in the lesson
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# define a callback because we want to stop when the accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy')>0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

# now we can define our new model layers
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')

# create the model
model = Model(pre_trained_model.input, x)

# compile the model
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# grab the images
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

# the validation data should not be augmented
test_datagen = ImageDataGenerator(rescale=1/255)

# training images
train_generator = train_datagen.flow_from_directory(
    os.path.join('../../../course1/week4/horse-or-human'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join('../../../course1/week4/validation-horse-or-human'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

callbacks = myCallback()

# run the model
history = model.fit(train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100, # 2000 training / 20 = 100
        epochs=3,
        validation_steps=50, # 1000 test / 20 = 50 
        verbose=2
)
