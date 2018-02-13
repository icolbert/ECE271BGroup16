import argparse
import numpy as np
from matplotlib import pyplot as plt

#from utils import *

def model_filter(x):
	x = str(x)
	return x.lower()

if __name__ == '__main__':
	parser = argparse.ArgumentParser('')

	parser.add_argument('-model',
		type=model_filter,
		default='mlp',
		#description='type of model'
		)

	parser.add_argument('-train',
		type=bool,
		default=False,
		#description='train?'
		)

	parser.add_argument('-path',
		type=str,
		default=None)

	args = parser.parse_args()

	if args.train:
		y = input('file path: ')
		print('train here')

	print(args.model)

	'''
	img = cv2.imread('/Users/josejoy/Desktop/ECE 271B Stat Learning /project/eq1.jpg',0)
	segments= extract_segments(img, 10)
	plt.imshow(segments[1])
	plt.show()
	'''