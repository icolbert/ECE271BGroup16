import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import numpy as np
import matplotlib.pyplot as plt


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class CNN(object):
    
    def __init__(self, x_dim=784, c_dim=19):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.name = 'CNN'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = tf.reshape(x, [-1, 28, 28, 1]) # reshapes it to 28x28 images
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

            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, self.c_dim, activation_fn=tf.sigmoid)
            return fc2

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class TrainModel:
    def __init__(self):
        self.data, self.labels = self.load_data()
        self.IN_DIM = self.data['train'].shape[1]
        self.NUM_CLASSES = self.labels['train'].shape[1]

        self.LEARNING_RATE = 1e-3
        self.DISPLAY_STEP = 100
        self.EPOCHS = 1000
        self.BATCHES = 50
        self.BATCH_SIZE = 500
        self.model = CNN(x_dim=self.IN_DIM, c_dim=self.NUM_CLASSES)

    def load_data(self):
        pass

    def train(self, inputs, labels):

        x = tf.placeholder(tf.float32, [None, self.IN_DIM])
        c = tf.placeholder(tf.float32, [None, self.NUM_CLASSES])

        pred = self.model(x)

        # Performance Metrics
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(c, 1)), tf.float32))

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=c))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(cost)

        self.acc = np.zeros((self.EPOCHS), dtype=float)
        self.tacc = np.zeros((self.EPOCHS), dtype=float)

        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            self.__minibatch_training(sess, optimizer, accuracy, cost, pred, x, c)
    
    def __minibatch_training(self, sess, optimizer, accuracy, cost, pred, x, c):

        mlog('Training {0}'.format(self.model.name))
        for epoch in range(self.EPOCHS):
            for batch in range(self.BATCHES):
                dx, dl = self.data['train'], self.labels['train']

                ridx = np.random.randint(dx.shape[0], size=self.BATCH_SIZE)
                xs, ls = dx[ridx,:], dl[ridx, :]

                sess.run(optimizer, feed_dict={x: xs, c:ls})
            sess.run(optimizer, feed_dict={x:self.data['val'], c:self.labels['val']})

            self.acc[epoch] = sess.run(accuracy, feed_dict={x:self.data['val'], c:self.labels['val']})
            self.tacc[epoch] = sess.run(accuracy, feed_dict={x:self.data['test'], c:self.labels['test']})

            if (epoch+1 % self.DISPLAY_STEP == 0):
                print('[Epoch {0}]'.format(epoch+1), end='\t')
                print('Cost: %.5f' % sess.run(cost, feed_dict={x: self.data['train'], c:self.labels['train']}), end='\t')
                print('Acc: %.3f' % 100.0*self.acc[epoch])

    def plot_results(self, save_path='results/'):

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