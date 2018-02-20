import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class CNN(object):
    
    def __init__(self, x_dim=784, c_dim=19):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.name = 'CNN'

    def __call__(self, x, reuse=True, dropout=0.25, is_training=True):
        with tf.variable_scope(self.name, reuse=reuse):
            x = tf.reshape(x, [-1, 28, 28, 1]) # reshapes it to 28x28 images
            '''conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )'''
            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.identity)
            conv1 = leaky_relu(conv1)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            '''conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )'''
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            conv2 = leaky_relu(conv2)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            conv2 = tcl.flatten(conv2)
            '''fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, self.c_dim, activation_fn=tf.sigmoid)'''
            fc1 = tf.layers.dense(conv2, 1024)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            out = tf.layers.dense(fc1, self.c_dim)
            return out

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class TrainModel:
    def __init__(self, **kwargs):
        self.data, self.labels = self.load_data(kwargs.pop('class_labels', False))
        self.IN_DIM = self.data['train'].shape[1]
        self.NUM_CLASSES = self.labels['train'].shape[1]

        self.LEARNING_RATE = 1e-3
        self.DISPLAY_STEP = 1
        self.EPOCHS = 500
        self.BATCHES = 50
        self.BATCH_SIZE = 500
        self.model = CNN(x_dim=self.IN_DIM, c_dim=self.NUM_CLASSES)

    def load_data(self, class_labels, train=0.85, val=0.15):
        print('Loading data...', end='\t\t\t')
        data = pd.read_csv('data/final_data.csv', header=None)
        labels = pd.read_csv('data/final_label.csv', header=None, names=['labels'])
        labels['labels'] = labels['labels'].map(class_labels)
        assert data.shape[0] == labels.shape[0]
        assert isinstance(train, float) and isinstance(val, float), "train and val must be of type float, not {0} and {1}".format(type(train), type(val))
        assert ((train + val) == 1.0), "train + val must equal 1.0"

        sidx = int(data.shape[0]*train)
        _data  = {'train': data.iloc[:sidx].as_matrix(),   'val': data.iloc[sidx+1:].as_matrix()}
        _labels= {'train': labels.iloc[:sidx].as_matrix(), 'val': labels.iloc[sidx+1:].as_matrix()}
        print('[Done]')

        assert (_data['train'].shape[0] == _labels['train'].shape[0]) and (_data['val'].shape[0] == _labels['val'].shape[0])
        return _data, _labels


    def train(self, save_path='/saved_models/'):
        if not os.path.exists(save_path):
              os.makedirs(save_path)

        x = tf.placeholder(tf.float32, [None, self.IN_DIM])
        c = tf.placeholder(tf.float32, [None, self.NUM_CLASSES])

        pred_train = self.model(x, reuse=False, is_training=True)
        pred_test = self.model(x, reuse=True, is_training=False)

        # Performance Metrics
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_train, 1), tf.argmax(c, 1)), tf.float32))
        taccuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_test, 1), tf.argmax(c, 1)), tf.float32))

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_train, labels=c))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(cost)

        self.acc = np.zeros((self.EPOCHS), dtype=float)
        self.tacc = np.zeros((self.EPOCHS), dtype=float)
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            self.__minibatch_training(sess, optimizer, accuracy, cost, pred_train, pred_test, x, c, taccuracy)
            
            sp = saver.save(sess, "/{0}/{1}.ckpt".format(self.model.name, save_path))
            print("Model saved in path: %s" % sp)
    
    def __minibatch_training(self, sess, optimizer, accuracy, cost, pred_train, pred_test, x, c, taccuracy):

        mlog('Training {0}'.format(self.model.name))
        for epoch in range(self.EPOCHS):
            for batch in range(self.BATCHES):
                dx, dl = self.data['train'], self.labels['train']

                ridx = np.random.randint(dx.shape[0], size=self.BATCH_SIZE)
                xs, ls = dx[ridx,:], dl[ridx, :]

                sess.run(optimizer, feed_dict={x: xs, c:ls})
            sess.run(optimizer, feed_dict={x:self.data['val'], c:self.labels['val']})

            self.acc[epoch] = sess.run(accuracy, feed_dict={x:self.data['train'], c:self.labels['train']})
            self.tacc[epoch] = sess.run(taccuracy, feed_dict={x:self.data['val'], c:self.labels['val']})

            if (epoch+1 % self.DISPLAY_STEP == 0):
                print('[Epoch {0}]'.format(epoch+1), end='\t')
                print('Cost: %.5f' % sess.run(cost, feed_dict={x: self.data['train'], c:self.labels['train']}), end='\t')
                print('Acc: %.3f' % self.acc[epoch])

    def plot_results(self, save_path='results/'):
        assert isinstance(save_path, str)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
    # Define the class labels
    class_labels = {str(x):x for x in range(10)}
    class_labels.update({'\\pi':10, '\\times':11, '\\%':12, '-':13, '/':14, '<':15, '>':16, '\\div':17, '+':18})
    
    # Initialize training the model
    model = TrainModel(class_labels=class_labels)
    model.train()
    model.plot_results()