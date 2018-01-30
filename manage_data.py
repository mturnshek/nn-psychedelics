import os
import numpy as np

"""
Assumes the model receives the 3 prior frames
to make its prediction for the following frame
"""

DATA_PATH = 'frame_sequence_data/' # give absolute path


def flatten(list_of_lists):
	return [val for sublist in list_of_lists for val in sublist]


def capture_segment(frames, i):
	x = np.array([frames[i+0], frames[i+1], frames[i+2]])
	y = np.array(frames[i+3])
	return x, y


def get_usable_dataset_from_file(filename):
	path = os.path.abspath(DATA_PATH + filename)
	frames = np.load(path)

	# create the dataset from this file
	# x is array of frames with length 3
	# y is a single frame
	X, Y = [], []
	for i in range(len(frames) - 4):
		x, y = capture_segment(frames, i)
		X.append(x)
		Y.append(y)

	return X, Y


def load_dataset():
	"""
	Load all files in the 'frame_sequence_data' directory into a single array.
	Assumes:
		they're all numpy array files with color frame data
		(in this case, they are arrays of shape (n, 110, 127, 3,))
		the shapes are the same past the first dimension
	"""
	files = os.listdir(os.path.abspath(DATA_PATH))
	X, Y = [], []
	for file in files:
		X_new, Y_new = get_usable_dataset_from_file(file)
		X.append(X_new)
		Y.append(Y_new)
		print('Loaded file ' + file)

	X, Y = flatten(X), flatten(Y)
	return np.array(X), np.array(Y)
