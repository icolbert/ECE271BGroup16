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
	eqn_full = [temp1, temp2, temp3, temp4]

	RFILE = open('cnn-results-ver{0}.txt'.format(args.data_ver), 'w')
	for eqns in eqn_full:
		for c in range(len(eqns)):
			eqn1 = eqns[c]
			segments= preprocess.extract_segments(eqn1, 30, reshape = 1, size = [28,28], 
												area=100, gray = True, dil = True,  ker = 2)
			cnn_pred = ''
			for i in range(len(segments)):
				temp = segments[i]
				'''plt.imshow(temp, cmap='gray')
				plt.show()'''
				temp = temp.reshape(1,-1)
				cnn_pred += label_class[model.predict(temp)[0]]+' '
			
			RFILE.write(cnn_pred+'\n')
			print(cnn_pred)
		print('[Done]')

	'''path = 'saved_models'
	adastage1_ver1,adadigits_ver1, adachars_ver1, rfmodel_ver1, MLP_single_ver1, CNN_ver_1 = preprocess.load_models(path,1,class_labels)
	#adastage1_ver2,adadigits_ver2, adachars_ver2, rfmodel_ver2, MLP_single_ver2, CNN_ver_2 = preprocess.load_models(path,2,class_labels)

	for eqns in eqn_full:
		for c in range(len(eqns)):
			# initialising strings to store output of model predictions
			rf_pred_ver1 = ''
			ada_pred_ver1 = ''
			mlp_pred_ver1 = ''
			#rf_pred_ver2 = ''
			#ada_pred_ver2 = ''
			#mlp_pred_ver2 = ''
			cnn_pred_ver1 = ''
			#cnn_pred_ver2 = ''

	#         print('\nEquation = ',c)
			eqn1 = eqns[c]
			# extract segments (digits/symbols) from each equation image
			segments= preprocess.extract_segments(eqn1, 40, reshape = 1, size = [28,28], 
												area=150, gray = True, dil = True,  ker = 1)

			# run prediction on each segment
			#plt.figure(figsize=(20,20))
			for i in range(len(segments)):
				temp = segments[i]
				temp = temp.reshape(1,-1)
				pred = preprocess.predict(temp, label_class, adastage1_ver1, 
											adadigits_ver1, adachars_ver1, rfmodel_ver1, MLP_single_ver1, CNN=CNN_ver_1)
				ada_pred_ver1 += pred[0] + ' '
				rf_pred_ver1 += pred[1] + ' '
				mlp_pred_ver1 += pred[2] + ' '
				cnn_pred_ver1 += pred[3] + ' '
				
				pred = preprocess.predict(temp, label_class, adastage1_ver2, 
											adadigits_ver2, adachars_ver2, rfmodel_ver2, MLP_single_ver2, CNN=CNN_ver_2)
				ada_pred_ver2 += pred[0] + ' '
				rf_pred_ver2 += pred[1] + ' '
				mlp_pred_ver2 += pred[2] + ' '
				cnn_pred_ver2 += pred[3] + ' '
				
			print('RF model_ver1 result : ',rf_pred_ver1)
			print('adaboost_ver1 2 stage model result : ',ada_pred_ver1)
			print('MLP_ver1 single stage model result : ',mlp_pred_ver1)
			print('CNN_ver1 results: ', cnn_pred_ver1)
			print('\nRF model_ver2 result : ',rf_pred_ver2)
			print('adaboost_ver2 2 stage model result : ',ada_pred_ver2)
			print('MLP single_ver2 stage model result : ',mlp_pred_ver2)
			print('CNN_ver2 result: ', cnn_pred_ver2)
			'''



