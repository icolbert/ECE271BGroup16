import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import datetime
import pandas as pd
import argparse
import numpy as np
import os

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class LogReg():
    def __init__(self, x_dim, c_dim):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.LEARNING_RATE = 1e-3
        self.EPOCHS = 1000
        self.BATCHES = 20
        self.BATCH_SIZE = 500
        self.name = 'LogReg'
    
    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            fc = tcl.fully_connected(
                x, self.c_dim,
                weights_initializer=tf.random_normal_initializer(stddev=2),
                weights_regularizer=tc.layers.l2_regularizer(1e-5),
                activation_fn=tf.identity
            )
            return fc, tf.nn.softmax(fc)
    
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class CNN(object):
    
    def __init__(self, x_dim, c_dim):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.LEARNING_RATE = 1e-5
        self.EPOCHS = 250
        self.BATCHES = 20
        self.BATCH_SIZE = 500
        self.name = 'CNN'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            x = tf.reshape(x, [-1, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            

            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            
            fc1 = tc.layers.fully_connected(
                tcl.flatten(conv2), 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            
            fc2 = tc.layers.fully_connected(
                fc1, self.c_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            
            return fc2, tf.nn.softmax(fc2)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class TrainModel:
    def __init__(self, model='cnn', **kwargs):
        assert isinstance(model, str), "the model type must be a string, not a {0}".format(type(model))
        
        self.data, self.labels = self.load_data(kwargs.pop('class_labels', False))
        self.IN_DIM = self.data['train'].shape[1]
        self.NUM_CLASSES = self.labels['train'].shape[1]
        self.DISPLAY_STEP = 10

        if model == 'log_reg':
            print('Using Logistic Regression')
            self.model = LogReg(x_dim=self.IN_DIM, c_dim=self.NUM_CLASSES)
        elif model == 'cnn':
            print('Using CNN')
            self.model = CNN(x_dim=self.IN_DIM, c_dim=self.NUM_CLASSES)

    def load_data(self, class_labels, train=0.85, val=0.15):
        print('Loading data...', end='\t\t\t')
        data = pd.read_csv('data/final_data.csv', header=None)
        data[data > 0] = 1
        labels = pd.read_csv('data/final_label.csv', header=None, names=['labels'])
        labels['labels'] = labels['labels'].map(class_labels)
        assert data.shape[0] == labels.shape[0]
        assert isinstance(train, float) and isinstance(val, float), "train and val must be of type float, not {0} and {1}".format(type(train), type(val))
        assert ((train + val) == 1.0), "train + val must equal 1.0"
        
        one_hot = pd.get_dummies(labels['labels'])
        sidx = int(data.shape[0]*train)
        _data  = {'train': data.iloc[:sidx].as_matrix(),   'val': data.iloc[sidx+1:].as_matrix()}
        _labels= {'train': one_hot.iloc[:sidx].as_matrix(), 'val': one_hot.iloc[sidx+1:].as_matrix()}
        print('[Done]')

        
        assert (_data['train'].shape[0] == _labels['train'].shape[0]) and (_data['val'].shape[0] == _labels['val'].shape[0])
        return _data, _labels
    
    def __call__(self, data, model_path='saved_models'):
        assert (data.shape[1] == self.IN_DIM), "Need to have the same dimension data and labels as what it was trained on"

        x = tf.placeholder(tf.float32, [None, self.IN_DIM])
        logits, probs = self.model(x)

        with tf.Session() as sess:
            #new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
            tf.train.Saver().restore(sess, '{0}/{1}/{1}.ckpt'.format(model_path, self.model.name))
            y = sess.run(probs, feed_dict={x: data})
        
        return np.argmax(y, axis=1)


    def train(self, save_path='saved_models/', from_model=False):
        if not os.path.exists('{0}/{1}/'.format(save_path, self.model.name)):
              os.makedirs('{0}/{1}/'.format(save_path, self.model.name))
        
        self.log_file = open('{0}/model_log.txt'.format(save_path), 'a')
        mlog('{0} Training {1}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), self.model.name), self.log_file)
        self.log_file.write(
            'Learning rate: {0}\nEpochs: {1}\nBatches: {2}\nBatch Size: {3}\n'.format(
            self.model.LEARNING_RATE, self.model.EPOCHS, self.model.BATCHES, self.model.BATCH_SIZE
            )
        )

        self.x = tf.placeholder(tf.float32, [None, self.IN_DIM])
        self.c = tf.placeholder(tf.float32, [None, self.NUM_CLASSES])

        self.logits, self.probs = self.model(self.x)
        self.correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.c, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.c))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model.LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(self.cost, var_list=self.model.vars)

        self.acc = np.zeros((self.model.EPOCHS), dtype=float)
        self.tacc = np.zeros((self.model.EPOCHS), dtype=float)
        self.saver = tf.train.Saver()
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if from_model: 
            print('Training from saved model')
            self.saver.restore(self.sess, "{0}/{1}/{1}.ckpt".format(save_path, self.model.name))
            self.__retraining()
        else: 
            self.__minibatch_training()
        
        sp = self.saver.save(self.sess, "{0}/{1}/{1}.ckpt".format(save_path, self.model.name))
        print('Model saved in path: {0}'.format(sp))
    
    def __minibatch_training(self):

        mlog('Training {0}'.format(self.model.name))
        for epoch in range(self.model.EPOCHS):
            for batch in range(self.model.BATCHES):
                if (batch+1 == self.model.BATCHES): self.sess.run(self.optimizer, feed_dict={self.x:self.data['val'], self.c:self.labels['val']})
                dx, dl = self.data['train'], self.labels['train']

                ridx = np.random.randint(dx.shape[0], size=self.model.BATCH_SIZE)
                xs, ls = dx[ridx,:], dl[ridx, :]

                self.sess.run(self.optimizer, feed_dict={self.x: xs, self.c:ls})
            self.sess.run(self.optimizer, feed_dict={self.x:self.data['val'], self.c:self.labels['val']})

            self.acc[epoch] = self.sess.run(self.accuracy, feed_dict={self.x:self.data['train'], self.c:self.labels['train']})
            self.tacc[epoch] = self.sess.run(self.accuracy, feed_dict={self.x:self.data['val'], self.c:self.labels['val']})

            print('[Epoch {0}]'.format(epoch+1), end='\t')
            print('Cost: %.5f' % self.sess.run(self.cost, feed_dict={self.x: self.data['train'], self.c:self.labels['train']}), end='\t')
            print('Acc: %.3f' % self.acc[epoch])
    
    def __retraining(self):

        mlog('Training {0} from Saved Model'.format(self.model.name))
        for epoch in range(50):
            for batch in range(self.model.BATCHES):
                if (batch+1 == self.model.BATCHES): self.sess.run(self.optimizer, feed_dict={self.x:self.data['val'], self.c:self.labels['val']})
                dx, dl = self.data['train'], self.labels['train']

                ridx = np.random.randint(dx.shape[0], size=self.model.BATCH_SIZE)
                xs, ls = dx[ridx,:], dl[ridx, :]

                self.sess.run(self.optimizer, feed_dict={self.x: xs, self.c:ls})

            self.sess.run(self.optimizer, feed_dict={self.x:self.data['val'], self.c:self.labels['val']})

            self.acc[epoch] = self.sess.run(self.accuracy, feed_dict={self.x:self.data['train'], self.c:self.labels['train']})
            self.tacc[epoch] = self.sess.run(self.accuracy, feed_dict={self.x:self.data['val'], self.c:self.labels['val']})

            print('[Epoch {0}]'.format(epoch+1), end='\t')
            print('Cost: %.5f' % self.sess.run(self.cost, feed_dict={self.x: self.data['train'], self.c:self.labels['train']}), end='\t')
            print('Acc: %.3f' % self.acc[epoch])


    def plot_results(self, save_path='results/'):
        assert isinstance(save_path, str)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.log_file.write(
            '\nFinal training accuracy: {0}\nFinal test accuracy: {1}\n'.format(
            self.acc[-1], self.tacc[-1]
            )
        )
        
        plt.plot(self.acc, color='red', linewidth=2, label='Training Accuracy')
        plt.plot(self.tacc, '--', color='black', linewidth=1, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc=4)

        plt.title('Accuracy')
        plt.savefig('{0}/{1}-acc.png'.format(save_path, self.model.name))

    

def mlog(message, File=None):
    assert isinstance(message, str)

    if File:
        File.write("=========================================\n")
        File.write(message)
        File.write("\n=========================================\n")
    else:
        print("\n=========================================")
        print(message)
        print("=========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    
    parser.add_argument('-model', default='cnn', type=str, help='Type of model to use')
    
    args = parser.parse_args()
    
    # Define the class labels
    class_labels = {str(x):x for x in range(10)}
    class_labels.update({'\\pi':10, '\\times':11, '\\%':12, '-':13, '/':14, '<':15, '>':16, '\\div':17, '+':18})
    
    # Initialize training the model
    model = TrainModel(model=args.model, class_labels=class_labels)
    model.train(from_model=False)
    model.plot_results()
    x = model.data['val'][0:2]
    print(model(x))