
# coding: utf-8

# In[2]:

import cv2
import numpy as np
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


# ## Image segmentation + Resampling


def rescale_segment( segment, size = [28,28], pad = 0 ):
    '''function for resizing (scaling down) images
    input parameters
    seg : the segment of image (np.array)
    size : out size (list of two integers)
    output 
    scaled down image'''
    if len(segment.shape) == 3 : # Non Binary Image
        import cv2
        # thresholding the image
        ret,segment = cv2.threshold(segment,127,255,cv2.THRESH_BINARY)
    m,n = segment.shape
    idx1 = list(range(0,m, (m)//(size[0]) ) )
    idx2 = list(range(0,n, n//(size[1]) )) 
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = segment[ idx1[i] + (m%size[0])//2, idx2[j] + (n%size[0])//2]
    return out

def extract_segments(img, pad=30, reshape = 0,size = [28,28], area = 150, threshold = 100, 
                     gray = False, dil = True, ker = 1) :
    '''function to extract individual chacters and digits from an image
    input paramterts
    img : input image (numpy array)
    pad : padding window size around segments (int)
    size : out size (list of two integers)
    reshape : if 1 , output will be scaled down. if 0, no scaling down
    area : Minimum area requirement for connected component detection
    thresh : gray scale to binary threshold value
    gray : if False, the segments returned will be binary, else will be gray scale
    dil : if True, performs dilation on segments, else erosion
    ker : dimesnion of kernel size for dilation / erosion
    Returns
    out : list of each segments (starting from leftmost digit)'''
    
    import cv2
    
    # thresholding the image
    ret,thresh1 = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    
    # Negative tranform gray levels (background becomes black) 
    thresh1 = 255 - thresh1
    img = 255 - img

    # connected component labelling 
    output = cv2.connectedComponentsWithStats(thresh1, 4)
    final = []
    temp2 = output[2]
    temp2 = temp2[temp2[:,4]>area]
    temp1 = np.sort( temp2[:,0] )
    kernel = np.ones( [ker, ker])

    for i in range(1,temp2.shape[0]):
        cord = np.squeeze( temp2[temp2[:,0] == temp1[i]] )
#         import pdb; pdb.set_trace()
#         print(cord)
    
        if gray == False:
            num = np.pad( thresh1[ cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2] ], pad,'constant')
        else :
            num = np.pad( img[ cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2] ], pad,'constant')

        if dil :
            num = cv2.dilate(num,kernel,iterations = 1)
        else :
            num = cv2.erode(num,kernel,iterations = 1)

        if reshape == 1:
            num = rescale_segment( num, size )
        final.append(num/255)
        
    return final

def load_data(class_labels, data_name, label_name, train=0.85, val=0.15):
    '''Function to Load data from .npy files and split them into training and validation sets
    Inputs
    class labels : Dictionary of class labels (dict)
    data_name : name of .npy data file, with path (str)
    label_name : name of .npy label file, with path (str)
    train : fraction of samples used in training set (float)
    val : fraction of samples used in training set (float)
    '''
    data = pd.DataFrame( np.load( data_name ) )
    labels = pd.DataFrame( np.load( label_name ) )
    
    labels = labels.rename(columns = {0:'labels'})
    
    labels['labels'] = labels['labels'].map(class_labels)
    assert data.shape[0] == labels.shape[0]
    assert isinstance(train, float)
    isinstance(val, float), "train and val must be of type float, not {0} and {1}".format(type(train), type(val))
    assert ((train + val) == 1.0), "train + val must equal 1.0"

    one_hot = pd.get_dummies(labels['labels'])
    sidx = int(data.shape[0]*train)
    _data  = {'train': data.iloc[:sidx].as_matrix(),   'val': data.iloc[sidx+1:].as_matrix()}
    _labels= {'train': one_hot.iloc[:sidx,:].as_matrix(), 'val': one_hot.iloc[sidx+1:,:].as_matrix()}

    assert (_data['train'].shape[0] == _labels['train'].shape[0])
    assert (_data['val'].shape[0] == _labels['val'].shape[0])
    return _data, _labels

def load_models(path, ver = 1):
    '''Function to Load trained models from given path, assuming the names of each models are known
    Inputs 
    path : path to the directory with stored models (str)
    ver : version of models loaded (1 = 20000 MNIST + HASY , 2 = 60000 MNIST + Kaggle)'''
    
    if ver == 1:
        with open(path+'Adaboost_stage1_ver1.1.sav','rb') as f:
            adastage1_ver1 = pickle.load(f)
        with open(path+'Adaboost_digits_ver1.1.sav','rb') as f:
            adadigits_ver1 = pickle.load(f)
        with open(path+'Adaboost_chars_ver1.1.sav','rb') as f:
            adachars_ver1 = pickle.load(f)
        with open(path+'RFmodel_ver1.1.sav','rb') as f:
            rfmodel_ver1 = pickle.load(f)
            
        # load json and create keras model
        json_file = open(path+'MLP_singlestage_ver1.1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        MLP_single_ver1 = model_from_json(loaded_model_json)
        # load weights into new model
        MLP_single_ver1.load_weights(path+"MLP_singlestage_ver1.1.h5")
        adam1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        MLP_single_ver1.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam1,metrics=['accuracy'] )
        print("Loaded MLP model from disk")

        json_file = open(path+'cnn_v1.json')
        loaded_model_json = json_file.read()
        json_file.close()
        cnn_v1 = model_from_json(loaded_model_json)
        cnn_v1.load_weights(path+'cnn_v1.h5')
        cnn_v1.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
        print('Loaded CNN-ver1 from disk')

        return adastage1_ver1,adadigits_ver1, adachars_ver1, rfmodel_ver1, MLP_single_ver1, cnn_v1
       
    if ver == 2:
        with open(path+'Adaboost_stage1_ver2.1.sav','rb') as f:
            adastage1_ver2 = pickle.load(f)
        with open(path+'Adaboost_digits_ver2.1.sav','rb') as f:
            adadigits_ver2 = pickle.load(f)
        with open(path+'Adaboost_chars_ver2.1.sav','rb') as f:
            adachars_ver2 = pickle.load(f)
        with open(path+'RFmodel_ver2.1.sav','rb') as f:
            rfmodel_ver2 = pickle.load(f)
            
        # load json and create keras model
        json_file = open(path+'MLP_singlestage_ver2.1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        MLP_single_ver2 = model_from_json(loaded_model_json)
        MLP_single_ver2.load_weights(path+"MLP_singlestage_ver2.1.h5")
        adam1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        MLP_single_ver2.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam1,metrics=['accuracy'] )
        print("Loaded model from disk")

        json_file = open(path+'cnn_v2.json')
        loaded_model_json = json_file.read()
        json_file.close()
        cnn_v2 = model_from_json(loaded_model_json)
        cnn_v2.load_weights(path+'cnn_v2.h5')
        cnn_v2.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
        print('Loaded CNN-ver2 from disk')

        return adastage1_ver2,adadigits_ver2, adachars_ver2, rfmodel_ver2, MLP_single_ver2, cnn_v2
    
    
def predict(temp, label_class, adastage1 = None,adadigits= None, adachars= None
            , rfmodel= None, MLP_single= None ):
    '''Function for using trained models to make prediction, and return predictions
    Inputs
    temp : image (segment) of digit / symbol
    label_class : dictionary of class labels
    adastage1 : Trained adaboost model stage1
    adadigits : Trained adaboost model for digits
    adachars :  Trained adaboost model for symbols
    rfmodel : Trained randomforest model
    MLP_single : Trained MLP model'''
    
    predictions = []
    if adastage1 :
        if adastage1.predict( temp )[0] == 1 : # stage 1 predicts digit
            ada_pred = label_class[adadigits.predict( temp )[0] ] 
        else : # stage 1 predicts symbol
            ada_pred = label_class[adachars.predict( temp )[0] ] 
        predictions.append(ada_pred)
        
    if rfmodel :        
        rf_pred = label_class[rfmodel.predict( temp )[0] ] + ' '
        predictions.append(rf_pred)
    
    if  MLP_single :
        mlp_pred = label_class[ np.argmax(  MLP_single.predict( temp ) ) ] 
        predictions.append(mlp_pred)
                           
    return predictions
    
    
    
    