from .segment_extraction import *

from .models import *


'''
Last update: 2/27/2018 -ic

Notes:
  - Need to update LoadModel() class to read in the equations, segment the images and then classify
  
'''
import tensorflow as tf 
import numpy as np


class LoadModel(TrainModel):
    def __init__(self,model='cnn', **kwargs):
        assert isinstance(model, str), "the model type must be a string, not a {0}".format(type(model))
        
        TrainModel.__init__(self, model=model, class_labels=kwargs.pop('class_labels', False), data_ver=kwargs.pop('data_ver', 1))

    def predict(self, data):
        assert (data.shape[1] == self.IN_DIM), "Need to have the same dimension data and labels as what it was trained on"

        x = tf.placeholder(tf.float32, [None, self.IN_DIM])
        logits, probs = self.model(x)

        with tf.Session() as sess:
            try:
                tf.train.Saver().restore(sess, '{0}/{1}/{1}.ckpt'.format(model_path, self.model.name))
                y = sess.run(probs, feed_dict={x: data})
            except Exception as e:
                print('Error: ', e)
                train_model = input('Should I train a new model? [y/n] ')
                if (train_model.lower() == 'y'):
                  self.train()
                  plot_acc = input('Should I plot the accuracy? [y/n] ')
                  if (plot_acc.lower() == 'y'):
                    self.plot_results()
                self.sess.run(self.probs, feed_dict={self.x: data})
        
        return np.argmax(y, axis=1)
