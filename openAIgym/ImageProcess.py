import numpy as np
from PIL import Image

def gray_scale(screen, resize=(84, 84)):
	screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
	screen = Image.fromarray(screen)
	screen = screen.resize(resize)
	screen = np.array(screen)
	screen = np.expand_dims(screen, axis=0)
	return screen















