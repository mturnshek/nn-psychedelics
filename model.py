import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, Activation, Reshape, Flatten, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

import numpy as np

from manage_data import load_dataset
import drugs


class FramePredictor:
	def __init__(self, activation_type='elu'):
		self.activation_type = activation_type
		self.batch_size = 64
		self.drug = 'none'
		self.X, self.Y = load_dataset()
		self.save_path = 'weights/frame_predictor_weights_dropout.hdf5'

		print(self.X.shape)
		print(self.Y.shape)

	def create_model(self):
		model = Sequential()
		model.add(Conv3D(32, (3, 3, 3), padding='same', input_shape=self.X.shape[1:]))
		model.add(Activation(self.activation_type))
		model.add(Dropout(0.25))
		model.add(Conv3D(32, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Dropout(0.25))
		model.add(Conv3D(32, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Dropout(0.25))
		model.add(Conv3D(32, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Dropout(0.25))
		model.add(Conv3D(1, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Reshape(self.Y.shape[1:]))
		model.compile(
			loss='mse',
			optimizer=RMSprop(),
		)
		self.model = model

	def train_model(self, epochs=200):
		checkpointer = ModelCheckpoint(
			filepath=self.save_path,
			verbose=1,
			save_best_only=True)
		self.model.fit(
			x=self.X,
			y=self.Y,
			validation_split=0.1,
			batch_size=self.batch_size,
			shuffle=True,
			epochs=epochs,
			callbacks=[checkpointer])

	def load_model(self):
		self.model.load_weights(self.save_path)
		self.clean_model = model

	def give_drug(self, drug):
		self.model = drug(self.model)

	def rehab(self):
		self.model = self.clean_model

	def predict(self, frames):
		return self.model.predict(frames)

