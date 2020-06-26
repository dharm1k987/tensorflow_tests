import tensorflow as tf
import os

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
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

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# training images
train_generator = train_datagen.flow_from_directory(
    os.path.join('./cats_and_dogs_filtered/train'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join('./cats_and_dogs_filtered/validation'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

# run the model

history = model.fit(train_generator,
                validation_data=validation_generator,
                steps_per_epoch=100, # 2000 training / 20 = 100
                epochs=15,
                validation_steps=50, # 1000 test / 20 = 50 
                verbose=2
)










import tensorflow as tf
import os

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
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

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# training images
train_generator = train_datagen.flow_from_directory(
    os.path.join('./cats_and_dogs_filtered/train'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join('./cats_and_dogs_filtered/validation'),
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

# run the model

history = model.fit(train_generator,
                validation_data=validation_generator,
                steps_per_epoch=100, # 2000 training / 20 = 100
                epochs=15,
                validation_steps=50, # 1000 test / 20 = 50 
                verbose=2
)