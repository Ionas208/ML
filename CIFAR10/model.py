import config

import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
import cv2


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def build_model(hp):
    model = keras.Sequential()
    input_shape = config.INPUT_SHAPE

    # VGG 1
    model.add(Conv2D(filters=hp.Int('conv1.1_filters', min_value=32, max_value=96, step=32), kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=hp.Int('conv1.2_filters', min_value=32, max_value=96, step=32), kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.7, step=0.1, default=0.2)))
    
    # VGG 2
    model.add(Conv2D(filters=hp.Int('conv2.1_filters', min_value=64, max_value=128, step=32), kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=hp.Int('conv2.2_filters', min_value=64, max_value=128, step=32), kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.7, step=0.1, default=0.2)))

    # VGG 3
    model.add(Conv2D(filters=hp.Int('conv3.1_filters', min_value=128, max_value=196, step=32), kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=hp.Int('conv3.2_filters', min_value=128, max_value=196, step=32), kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=hp.Float('dropout3', min_value=0.1, max_value=0.7, step=0.1, default=0.2)))

    # Dense
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout4', min_value=0.1, max_value=0.7, step=0.1, default=0.2)))
    model.add(Dense(units=config.NUM_CLASSES, activation='softmax'))

    # Compile
    lr = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.0001, default=0.001)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model