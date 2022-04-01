import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import config
import model
import utils

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import keras_tuner as kt
from sklearn.metrics import classification_report
import numpy as np
import cv2
(x_train, y_train), (x_test, y_test) = utils.load_data()

es = EarlyStopping(
	monitor="val_loss",
	patience=config.EARLY_STOPPING_PATIENCE,
	restore_best_weights=True)

tuner = kt.BayesianOptimization(
		model.build_model,
		objective="val_accuracy",
		max_trials=10,
		seed=42,
		directory=config.OUTPUT_PATH,
		project_name="cifar10_tuning")

tuner.search(
	x=x_train, y=y_train,
	validation_data=(x_test, y_test),
	batch_size=config.BS,
	callbacks=[es],
	epochs=config.EPOCHS
)

bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters: ", bestHP)