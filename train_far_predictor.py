import numpy as np

from model import FramePredictor


frame_predictor = FramePredictor()
frame_predictor.load_data()
frame_predictor.create_model()

X, Y = frame_predictor.X, frame_predictor.Y
given_frames = np.copy(X[0]) # start with all real frames

def update_given_frames(i, prediction):
	given_frames[0] = np.copy(X[i][0])
	given_frames[1] = np.copy(given_frames[2])
	given_frames[2] = np.copy(prediction)
	return given_frames

frame_predictor.model.save_weights(frame_predictor.save_path)

print(len(X))
epochs = 200
for epoch in range(epochs):
	for i in range(len(X)):
		frame_predictor.model.train_on_batch(
			x=np.array([given_frames]),
			y=np.array([Y[i]]))
		prediction = frame_predictor.predict(given_frames)
		update_given_frames(i, prediction)
	print("epoch", epoch+1, "200")
	frame_predictor.model.save_weights(frame_predictor.save_path)