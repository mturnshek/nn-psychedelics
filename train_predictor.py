from model import FramePredictor

frame_predictor = FramePredictor()
frame_predictor.load_data()
frame_predictor.create_model()
frame_predictor.train_model()