import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, Activation, Reshape, Flatten, Dense, Conv2DTranspose
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

import numpy as np

from manage_data import load_dataset
import drugs

class FramePredictor:
	def __init__(self, activation_type='elu', load_weights=False):
		self.activation_type = activation_type
		self.batch_size = 64
		self.save_path = 'weights/frame_predictor_weights.hdf5'

		# drugs settings
		self.on_shrooms = False

		if load_weights:
			self.load_data()
			self.create_model()
			self.load_model()

	def load_data(self):
		self.X, self.Y = load_dataset()
		self.predicted_frame_buffer = np.zeros(self.X[0].shape)

	def add_frame_to_predicted_frame_buffer(self, frame):
		self.predicted_frame_buffer[0] = self.predicted_frame_buffer[1]
		self.predicted_frame_buffer[1] = self.predicted_frame_buffer[2]
		self.predicted_frame_buffer[2] = frame

	def create_model(self):
		model = Sequential()
		model.add(Conv3D(64, (3, 3, 3), padding='same', input_shape=self.X.shape[1:]))
		model.add(Activation(self.activation_type))
		model.add(Conv3D(64, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Conv3D(64, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Conv3D(64, (3, 3, 3), padding='same'))
		model.add(Activation(self.activation_type))
		model.add(Conv3D(3, (3, 3, 3), strides=(3, 1, 1), padding='same'))
		model.add(Reshape(self.Y.shape[1:]))
		model.add(Activation(self.activation_type))
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

	def give_drug(self, drug):
		self = drug(self)

	def rehab(self):
		self.create_model()
		self.load_model()
		self.on_shrooms = False

	def predict(self, frames):
		prediction = self.model.predict(np.array([frames]))[0]
		self.add_frame_to_predicted_frame_buffer(prediction)
		return prediction

