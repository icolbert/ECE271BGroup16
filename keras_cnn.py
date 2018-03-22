import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sys
import argparse

# Define the class labels
class_labels = {str(x):x for x in range(10)}
class_labels.update({'+':10, 'times':11, '-':12 })
label_class = dict( zip(class_labels.values(), class_labels.keys() ))

def bool_filter(x):
    if x.lower() in ['true', 't', 'yes', 'y']: return True
    elif x.lower() in ['no', 'false', 'n', 'f']: return False
    else: raise NameError('True or False question bro...')

class DataLoader():

    def __init__(self, train=True, **kwargs):

        self.ver = kwargs.pop('data_ver', 1)
        self.data, self.labels = self.load_data(kwargs.pop('class_labels', {str(x):x for x in range(10)}))
        if train:
            self.data = self.data['train']
            self.labels = self.labels['train']
        else:
            self.data = self.data['val']
            self.labels = self.labels['val']
        self.transform = kwargs.pop('transform', None)

        #print(self.labels.shape)
        assert self.data.shape[0] == self.labels.shape[0], "Dimension mismatch"

    def load_data(self, class_labels, train=0.90, val=0.10):
        '''
        Function to Load data from .npy files and split them into training and validation sets
        Inputs
        class labels : Dictionary of class labels (dict)
        data_name : name of .npy data file, with path (str)
        label_name : name of .npy label file, with path (str)
        train : fraction of samples used in training set (float)
        val : fraction of samples used in training set (float)
        '''
        data = pd.DataFrame(np.load('data/training-data/data_ver{0}.npy'.format(self.ver)))
        labels = pd.DataFrame(np.load('data/training-data/labels_ver{0}.npy'.format(self.ver)))
        
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
    
    def __len__(self):
        return self.data.shape[0]

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    
    parser.add_argument('-model', default=3, type=int, help='Type of model to use')
    parser.add_argument('-ver', default=1, type=int, help='version of data to use')
    parser.add_argument('-verbose', default=False, type=bool_filter, help='show a lot more outputs? [y/n]')
    
    args = parser.parse_args()

    # input image dimensions
    img_x, img_y = 28, 28
    # load the MNIST data set, which already splits into train and test sets for us
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    training_data_loader = DataLoader(train=True, class_labels=class_labels, data_ver=1)
    test_data_loader = DataLoader(train=False, class_labels=class_labels, data_ver=1)

    x_train, y_train = training_data_loader.data, training_data_loader.labels
    x_test, y_test = test_data_loader.data, test_data_loader.labels

    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if args.verbose:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices - this is for use in the
    #y_train = keras.utils.to_categorical(y_train)
    #y_test = keras.utils.to_categorical(y_test)
    if args.verbose:
        print('y_train shape: ', y_train.shape)
        print('y_test shape: ', y_test.shape)
        print(y_train.shape[0], 'train samples')
        print(y_test.shape[0], 'test samples')

    batch_size = 256
    num_classes = y_train.shape[1]
    epochs = 25

    
    if args.model == 1:
        model = Sequential()
        model.add(Conv2D(256, kernel_size=(3, 3),input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1000 ))
        model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(Dropout(0.1))

        model.add(Dense(256 ))
        model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(Dropout(0.1))

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(Dropout(0.1))

        model.add(Dense(num_classes, activation='softmax'))

    if args.model == 2:
        model = Sequential()
        model.add(Conv2D(96, kernel_size=(3, 3), input_shape=input_shape))
        model.add(keras.layers.ELU(alpha=0.1))
        model.add(Conv2D(96, kernel_size=(1, 1)))
        model.add(keras.layers.ELU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(192, kernel_size=(3, 3)))
        model.add(keras.layers.ELU(alpha=0.1))
        model.add(Conv2D(192, kernel_size=(1, 1)))
        model.add(keras.layers.ELU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(192, kernel_size=(1, 1)))
        model.add(keras.layers.ELU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(AveragePooling2D(pool_size=(3,3)))

        model.add(Flatten())
        model.add(Dense(256 ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(num_classes, activation='softmax'))

    if args.model == 3:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),input_shape=input_shape))
        model.add(BatchNormalization())
        model.add( keras.layers.LeakyReLU(alpha=0.1)  )
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add( keras.layers.LeakyReLU(alpha=0.1)  )
        model.add(Dropout(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1000 ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(256 ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(num_classes, activation='softmax'))

    adam1 = keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam1,metrics=['accuracy'] )

    model.summary()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    validation_data=(x_test, y_test))

    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # summarize history for accuracy
    plt.figure(figsize=(8,4))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('CNN model ver{0} accuracy'.format(args.ver),fontsize = 14)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.ylabel('accuracy',fontsize = 14)
    plt.xlabel('epochs',fontsize = 14)
    plt.legend(['train', 'val'], loc='upper left',fontsize = 12)
    plt.savefig('cnn-m{1}-ver{0}-acc.png'.format(args.ver, args.model))

    # summarize history for loss
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN model ver{0} loss'.format(args.ver),fontsize = 14)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.ylabel('loss',fontsize = 14)
    plt.xlabel('epochs',fontsize = 14)
    plt.legend(['train', 'val'], loc='upper right', fontsize = 12)
    plt.savefig('cnn-m{1}-ver{0}-loss.png'.format(args.ver, args.model))

    mjson = model.to_json()
    with open('saved_models/cnn_m{1}_v{0}.json'.format(args.ver, args.model), 'w') as f:
        f.write(mjson)
    model.save_weights("saved_models/cnn_m{1}_v{0}.h5".format(args.ver, args.model))
    print("Saved model to disk")

