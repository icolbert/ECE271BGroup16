import argparse
import numpy as np
import pandas as pd

from utils import *

def model_filter(x):
	x = str(x)
	return x.lower()

if __name__ == '__main__':
	parser = argparse.ArgumentParser('')

	parser.add_argument('-model',
		type=model_filter,
		default='cnn'
		)

	parser.add_argument('-train',
		type=bool,
		default=False
		)

	parser.add_argument('-path',
		type=str,
		default=None
		)

	args = parser.parse_args()

	class_labels = {str(x):x for x in range(10)}
	class_labels.update({'\\pi':10, '\\times':11, '\\%':12, '-':13, '/':14, '<':15, '>':16, '\\div':17, '+':18})
	label_class = dict(zip(class_labels.values(), class_labels.keys()))

	eqns = np.load('data/equation_data/Equations_images_1.npy')
	labels = pd.read_excel('data/equation_data/Equations_labels.xlsx', sheet_name='Equations_images_1_labels')
	eqn1, label1 = eqns[0], labels.iloc[0].values
	
	model = TrainModel(model=args.model, class_labels=class_labels)
	for c in range(len(eqns)):
		print('\nEquation = ',c)
		segments= extract_segments(eqns[c], 30, reshape=1, size=[28,28])
		pred = ''
		for segment in segments:
			pred += label_class[model(segment.reshape(1,-1))[0]] + ' '
		print('Model result: ', pred)

