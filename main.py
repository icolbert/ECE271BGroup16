import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import LoadModel
from utils import preprocess

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
	
	parser.add_argument('-data_ver',
		type=int,
		default=1
	)

	args = parser.parse_args()
	
	class_labels = {str(x):x for x in range(10)}
	class_labels.update({'+':10, 'times':11, '-':12 })
	label_class = dict( zip(class_labels.values(), class_labels.keys() ))
	model = LoadModel(model=args.model, class_labels=class_labels, data_ver=args.data_ver)
	x = model.data['val'][0:2]
	print(model(x))

	path1 = 'data/equation-data/'
	temp1 = np.load(path1+'Equations_images_1.npy')
	temp2 = np.load(path1+'Equations_images_2.npy')
	temp3 = np.load(path1+'Equations_images_3.npy')
	temp4 = np.load(path1+'Equations_images_4.npy')
	eqn_full = [temp1,temp2,temp3,temp4]

	for eqns in eqn_full:
		for c in range(len(eqns)):
			# initialising strings to store output of model predictions
			rf_pred_ver1 = ''
			ada_pred_ver1 = ''
			mlp_pred_ver1 = ''
			rf_pred_ver2 = ''
			ada_pred_ver2 = ''
			mlp_pred_ver2 = ''
			cnn_pred_ver1 = ''

	#         print('\nEquation = ',c)
			eqn1 = eqns[c]
			# extract segments (digits/symbols) from each equation image
			segments= preprocess.extract_segments(eqn1, 40, reshape = 1, size = [28,28], 
												area=150, gray = True, dil = True,  ker = 1)

			# run prediction on each segment
			#plt.figure(figsize=(20,20))
			'''plt.imshow(eqn1, cmap='gray')
			plt.show()
			print(len(segments))'''
			for i in range(len(segments)):
				temp = segments[i]
				temp = temp.reshape(1,-1)
				cnn_pred_ver1 += label_class[model.predict(temp)[0]]
			
			print(cnn_pred_ver1)



					
			#plt.show()