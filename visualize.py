import pygame
import numpy as np
import sys
import time
import scipy.misc
import keyboard

from model import FramePredictor

# Initialize FramePredictor model
frame_predictor = FramePredictor(load_weights=True)
X, Y = frame_predictor.X, frame_predictor.Y # our dataset

# Initialize frame data display
pygame.init()
size = width, height = 508, 440
screen = pygame.display.set_mode(size)
pygame.event.set_grab(False)

speed = 0.05 # show 20fps

i = 0
while True:
	frames = X[i]
	if frame_predictor.predict_next_frame_mode:
		frames = frame_predictor.predicted_frame_buffer
	prediction = frame_predictor.predict(frames)
	prediction = scipy.misc.imresize(prediction, (440, 508))
	prediction = np.swapaxes(prediction, 0, 1)
	screen.blit(pygame.surfarray.make_surface(prediction), (0, 0))
	pygame.display.flip()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

		if keyboard.is_pressed('1'):
			frame_predictor.predict_next_frame_mode = True

		if keyboard.is_pressed('2'):
			frame_predictor.predict_next_frame_mode = False

	i += 1
	if i == len(X):
		i = 0
	time.sleep(speed)
