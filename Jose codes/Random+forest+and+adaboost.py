
# coding: utf-8

# In[170]:

import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import pickle
import preprocess


# In[171]:

# Creating dictionary of labels
class_labels = {str(x):x for x in range(10)}
class_labels.update({'+':10, 'times':11, '-':12 })
label_class = dict( zip(class_labels.values(), class_labels.keys() ))

# Loading data from .npy file and spliting into training and validation sets
path = '/Users/josejoy/Desktop/ECE 271B Stat Learning /project/Training Data/'
data, labels = preprocess.load_data(class_labels, path+'data_ver2.npy' , path+'labels_ver2.npy'
                                   , train = 0.85 , val = 0.15)


# In[173]:

idx = np.random.randint(data['train'].shape[0])
plt.imshow(data['train'][idx,:].reshape(28,28),cmap = 'gray')
plt.colorbar()
plt.show()
print('Label = ',label_class[ np.argmax(labels['train'][idx,:])])



# ## Random Forest Model training

# In[163]:

get_ipython().run_cell_magic('time', '', "train_label = np.argmax(labels['train'], axis = 1)\nRFmodel = RandomForestClassifier(max_depth=4, n_estimators=2000, class_weight='balanced' )\nRFmodel.fit(data['train'],train_label)\npred = RFmodel.predict(data['val'])\ntest_label = np.argmax(labels['val'], axis = 1)\nerror1 = np.sum([pred!=test_label])*100/test_label.shape[0]\nprint( error1 )\nplt.imshow( RFmodel.feature_importances_.reshape([28,28]) ,cmap = 'gray' )\nplt.show()\n\n## Saving trained model\n# filename = './trained models/RFmodel_ver1.1.sav'\n# pickle.dump(RFmodel, open(filename, 'wb'))")


# ## AdaBoost Stage 1 ( BInary classifer for Digits - Character classification)

# In[165]:

get_ipython().run_cell_magic('time', '', "\n# converting traing labels to binary, for  digits / characters\nbin_label = np.argmax(labels['train'], axis = 1)\nbin_label[bin_label<10] = 1 # digits have label 1\nbin_label[bin_label>=10] = -1 # symbols have label -1\n\nstage1 = AdaBoostClassifier( n_estimators=250)\nstage1.fit(data['train'],bin_label )\npred1 = stage1.predict(data['val'])\n\n# converting testing labels to binary, for  digits / characters\ntest = np.argmax(labels['val'], axis = 1)\ntest[test<10] = 1\ntest[test>=10] = -1\n\nprint( 'Stage1 error = ',sum( pred1!=test )*100/ test.shape[0])\nplt.imshow( stage1.feature_importances_.reshape([28,28]) ,cmap = 'gray' )\nplt.show()\n\n## Saving trained model\n# filename = './trained models/Adaboost_stage1_ver1.1.sav'\n# pickle.dump(stage1, open(filename, 'wb'))")


# ## AdaBoost Stage 2 (Digits classifier and symbol classifier)

# In[167]:

# Creating labels for digits classifier, -1 is the class for any non digit symbol 
label1 = np.argmax(labels['train'],axis=1)
label1[label1>=10] = -1

# Creating labels for symbol classifier, -1 is the class for any non symbol digit
label2 = np.argmax(labels['train'],axis=1)
label2[label2<10] = -1


# contains binary classifiers (one vs all) for each digit
digits = AdaBoostClassifier( n_estimators=500, learning_rate=0.1) 
digits.fit(data['train'][label1!=-1],label1[label1!=-1])

# contains binary classifiers (one vs all) for each digit
chars =  AdaBoostClassifier( n_estimators=500, learning_rate=0.1) 
chars.fit(data['train'][label2!=-1],label2[label2!=-1])

pred1 = stage1.predict(data['val'])
pred_d = digits.predict(data['val'])
pred_c = chars.predict(data['val'])
test_full = np.argmax(labels['val'],axis=1)
pred_d[pred1==-1] = -1
pred_c[pred1==1] = -1
pred = np.column_stack((pred_d,pred_c))
predx = np.max(pred,axis= 1)
print( 'Stage2 error = ',sum( predx!=test_full )*100/ test_full.shape[0])

## Saving data
# filename = './trained models/Adaboost_digits_ver1.1.sav'
# pickle.dump(digits, open(filename, 'wb'))
# filename = './trained models/Adaboost_chars_ver1.1.sav'
# pickle.dump(chars, open(filename, 'wb'))


# In[169]:

plt.imshow( digits.feature_importances_.reshape([28,28]) ,cmap = 'gray' )
plt.show()

