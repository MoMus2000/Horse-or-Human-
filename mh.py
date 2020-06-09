from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import json
import tensorflow as tf
import os

# Directory with our training horse pictures
train_horse_dir = os.path.join('/Users/a./Desktop/course/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/Users/a./Desktop/course/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/Users/a./Desktop/course/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/Users/a./Desktop/course/validation-horse-or-human/humans')




model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer = RMSprop(lr=0.001),metrics = ['acc'])

train_dir = "/Users/a./Desktop/course/horse-or-human"
test_dir = "/Users/a./Desktop/course/validation-horse-or-human"
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(300,300),
batch_size = 128, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(300,300),batch_size=32,class_mode='binary')


model.fit(
train_generator,steps_per_epoch = 8,epochs = 3, validation_data = test_generator,
validation_steps =8
)

model.save('/Users/a./Desktop/course')
