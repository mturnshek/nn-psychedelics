from keras.models import Sequential
from keras.layers import ELU, Activation

import numpy as np

"""
functions to be applied to a model
"""

#####################
# utility functions #
#####################


def layer_copies(model):
	""" 
	Copy layers from a sequential model returns the layers in order.
	They will have the same weights and configuration,
	but are not attached to a model.
	"""
	layer_copies = []

	layers = frame_predictor.model.layers
	for layer in layers:
		layer.get_config()


##############
# hard drugs #
##############

def caffeine(frame_predictor, rate=1.5):
	"""
	Given a model, return a copy with
	elu activation layers given 5% (rate) higher alpha
	"""
	drugged_model = Sequential()

	for layer in frame_predictor.model.layers:
		layer_config = layer.get_config()
		if 'activation' in layer_config:
			if layer_config == 'elu':
				drugged_model.add(ELU(alpha=rate))
			else:
				drugged_model.add(layer)
		else:
			drugged_model.add(layer)

	frame_predictor.model = drugged_model


def ecstasy(model, rate=1.05):
	"""
	Given a model, return a copy with
	all biases relaxed by 5% (rate)
	"""
	pass


def lsd(frame_predictor):
	"""
	swap weights between two layers
	"""
	swap_1 = 2
	swap_2 = 4
	
	weights_1 = frame_predictor.model.layers[swap_1].get_weights()
	weights_2 = frame_predictor.model.layers[swap_2].get_weights()

	frame_predictor.model.layers[swap_2].set_weights(weights_1)
	frame_predictor.model.layers[swap_1].set_weights(weights_2)


def shrooms(frame_predictor):
	frame_predictor.on_shrooms = True
