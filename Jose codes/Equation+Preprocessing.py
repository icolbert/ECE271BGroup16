
# coding: utf-8

# In[5]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
import PIL
import tensorflow as tf
from preprocess import rescale_segment as rescale_segment
from preprocess import extract_segments as extract_segments


# ## Equation manipulation

# In[25]:

img = cv2.imread('/Users/josejoy/Desktop/ECE 271B Stat Learning /project/Equation data/eqn4.jpg',0)
plt.imshow(img,cmap = 'gray')
plt.show() 
image = [] ## for eqn_im_1
for i in range(4):
    for j in range(4):
        x1 = 370*i ; x2 = x1+350; y1 = 500*j ; y2 = y1 + 500
        temp = img[x1:x2,y1:y2];
        kernel = np.ones( [3,3])
        temp = cv2.erode(temp,kernel,iterations = 1)
        image.append(temp)
        plt.imshow(temp,cmap = 'gray')
        plt.show() 


# ## Testing out extracting_segments, to see if images are proper

# In[27]:

for i in range(len(image)):
    im1 = image[i]
    segments= extract_segments(im1, 30, reshape = 1, size = [28,28], 
                               threshold = 40, area = 100, ker = 1, gray = True)
    plt.figure(figsize=[15,15])
    plt.subplot(181)
    plt.imshow(im1,cmap = 'gray')
    for j in range(len(segments)):
        plt.subplot(182+j)
        plt.imshow(segments[j],cmap = 'gray')
    plt.show()


# In[28]:

# ## Saving each equation as numpy file
# np.save('./Equation data/Equations_images_4.npy',np.array(image))

