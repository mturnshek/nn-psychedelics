import numpy as np
import pygame
import sys
import scipy.misc
import time
import keyboard

from model import FramePredictor
from drugs import shrooms


frame_predictor = FramePredictor(load_weights=True)
X, Y = frame_predictor.X, frame_predictor.Y # our dataset
given_frames = np.copy(X[0]) # start with all real frames

def update_given_frames(i, prediction):
	given_frames[0] = X[i][0]
	given_frames[1] = given_frames[2]
	given_frames[2] = prediction

# Initialize frame data display
pygame.init()
size = width, height = 508, 440
screen = pygame.display.set_mode(size)
pygame.event.set_grab(False)

speed = 0.05 # show 20fps

i = 0
while True:
	frames = given_frames
	if frame_predictor.on_shrooms:
		frames = frame_predictor.predicted_frame_buffer
	prediction = frame_predictor.predict(frames)
	update_given_frames(i, prediction)
	prediction = scipy.misc.imresize(prediction, (440, 508))
	prediction = np.swapaxes(prediction, 0, 1)
	screen.blit(pygame.surfarray.make_surface(prediction), (0, 0))
	pygame.display.flip()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

		if keyboard.is_pressed('1'):
			frame_predictor.rehab()

		if keyboard.is_pressed('4'):
			frame_predictor.give_drug(shrooms)

	i += 1
	if i == len(X):
		i = 0
	time.sleep(speed)