
# coding: utf-8

# In[1]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import preprocess
import pickle 
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import keras


# In[7]:

# creating dictionary for labels
class_labels = {str(x):x for x in range(10)}
class_labels.update({'+':10, 'times':11, '-':12 })
label_class = dict( zip(class_labels.values(), class_labels.keys() ))

path1 = '/Users/josejoy/Desktop/ECE 271B Stat Learning /project/Data/Equation data/'
eqns = np.load(path1+'Equations_images_2.npy')



# ## Loading saved models for prediction

# In[9]:

path2 = '/Users/josejoy/Desktop/ECE 271B Stat Learning /project/trained models/'
with open(path2+'Adaboost_stage1_ver2.1.sav','rb') as f:
    adastage1 = pickle.load(f)
with open(path2+'Adaboost_digits_ver2.1.sav','rb') as f:
    adadigits = pickle.load(f)
with open(path2+'Adaboost_chars_ver2.1.sav','rb') as f:
    adachars = pickle.load(f)
with open(path2+'RFmodel_ver2.1.sav','rb') as f:
    rfmodel = pickle.load(f)
    
# load json and create keras model
json_file = open('./trained models/MLP_singlestage_ver2.1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
MLP_single = model_from_json(loaded_model_json)
# load weights into new model
MLP_single .load_weights("./trained models/MLP_singlestage_ver2.1.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
adam1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
MLP_single.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam1,metrics=['accuracy'] )



# In[11]:

# Evaluating each . equation image

for c in range(len(eqns)):
    # initialising strings to store output of model predictions
    rf_pred = ''
    ada_pred = ''
    mlp_pred = ''
    
    print('\nEquation = ',c)
    eqn1 = eqns[c]
    # extract segments (digits/symbols) from each equation image
    segments= preprocess.extract_segments(eqn1, 20, reshape = 1, size = [28,28], 
                                          area=200, gray = True, dil = True,  ker = 2)
    
    # run prediction on each segment
    for i in range(len(segments)+1):
        if i ==0:
            plt.imshow(eqn1,cmap = 'gray')
            plt.figure(figsize=(18,20))
        else :
            # plot each segment
            plt.subplot(191+i)
            temp = segments[i-1]
            plt.imshow(temp,cmap = 'gray')
            temp = temp.reshape(1,-1)
            
            # Random forest model prediction
            rf_pred += label_class[rfmodel.predict( temp )[0] ] + ' '
            
            # Adaboost model prediction 
            if adastage1.predict( temp )[0] == 1 : # stage 1 predicts digit
                ada_pred += label_class[adadigits.predict( temp )[0] ] + ' '
            else : # stage 1 predicts symbol
                ada_pred += label_class[adachars.predict( temp )[0] ] + ' '
            
            # MLP prediction
            mlp_pred += label_class[ np.argmax(  MLP_single.predict( temp ) ) ] + ' '
                 
    plt.show()
    print('RF model result : ',rf_pred)
    print('adaboost 2 stage model result : ',ada_pred)
    print('MLP single stage model result : ',mlp_pred)

