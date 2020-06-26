import tensorflow as tf
import os

DESIRED_ACCURACY = 0.999

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>DESIRED_ACCURACY):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

# initially it's a 150x150 colour image
# 3 convulation layers
# 1 output layer
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

# combpile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 3
validation_split = 0.2
e_in_train = 80 * (1 - validation_split)
e_in_valid = 80 * validation_split


train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# grab all the data from the directory
train_generator = train_datagen.flow_from_directory(
    os.path.join('./happy-or-sad'),  
    target_size=(150, 150), # the image size is 150x150
    batch_size=batch_size, # since there are only 80 images in total, we will take batch size of 10
    class_mode='binary', # happy or sad
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join('./happy-or-sad'),  
    target_size=(150, 150), # the image size is 150x150
    batch_size=batch_size, # since there are only 80 images in total, we will take batch size of 10
    class_mode='binary', # happy or sad
    subset='validation'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=e_in_train // batch_size, # 80 / batch size = 80 / 10 = 8
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=e_in_valid // batch_size,
    callbacks=[callbacks]
)