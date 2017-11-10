import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, Activation
from keras.optimizers import RMSProp

import numpy as np


class FramePredictor:
	def __init__(self, n_input_frames=3, activation_type='relu'):
		self.n_input_frames = n_input_frames
		self.activation_type = activation_type
		self.batch_size = 64
		self.name = f'{self.n_input_frames}_i_f_{self.activation_type}.h5'
		self.drug = 'none'

	def create_model(self):
		model = Sequential()
		model.add(Conv3D(128, (1, 3, 3), padding='same', input_shape=self.x_train[1:]))
		model.add(Activation(self.activation_type))
		model.add(Conv3D(128, (1, 3, 3)))
		model.add(Activation(self.activation_type))
		model.add(Conv3D(128, (1, 3, 3)))
		model.add(Activation(self.activation_type))		
		model.compile(
			loss='mse',
			optimizer=RMSProp(),
		)
		self.model = model

	def train_model(self, epochs=200):
		self.model.fit(
			x=self.x_train,
			y=self.y_train,
			validation_split=0.1,
			batch_size=self.batch_size,
			epochs=epochs)
		self.model.save_weights('weights/' + self.name)

	def load_model(self):
		self.model.load_weights(self.name)
		self.clean_model = model

	def give_drug(self, drug):
		self.model = drug(self.model)

	def rehab(self):
		self.model = self.clean_model

	def predict(self, frames):
		pass

