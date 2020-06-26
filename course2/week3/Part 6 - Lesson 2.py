import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# transfer learning is when you take somebody elses model and add onto it
# usually the convolutions are already trained, and you only train the dense layers

# here are the pretrained weights for the inception neural network
# this is a snapshot of the model after it has been trained
local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# keras offers the model as a plugin, but here we will specify the input shape
# the weights are also set to None, because we want the weights to be the pretrained ones
# so we don't have to train the model again to acquire them
# include_top is set to false because we don't want to include the final dense layers
# we don't want this because we have our own input_shape
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None,
)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# now we can take whichever layer we want to be our 'last' layer
# this is still a convoluational layer
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# now define new model layers, taking output from inception models mixed7 layer
x = layers.Flatten()(last_output) # flatten the input, which is the output from inception
x = layers.Dense(1024, activation='relu')(x) # add dense hidden layer of 1024 neurons
x = layers.Dropout(0.2)(x)
'''
After the model was run, we noticed that the validation accuracy actually decreases
after each epoch. This is different from when the validation accuracy simply
levelled off at a maximum. This is still overfitting, and in this case, when it decreases,
we can try adding a dropout, which basically means that don't consider the output from
20% of the neurons
'''
x = layers.Dense(1, activation='sigmoid')(x) # add final output layer

# create the model
model = Model(pre_trained_model.input, x)

# compile the model
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# preprocess the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

