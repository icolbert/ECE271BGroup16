import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 13)
        
    def forward(self, x):
        out = self.layer1(x).double()
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class EqDataLoader(Dataset):

    def __init__(self, train=True, **kwargs):

        self.ver = kwargs.pop('data_ver', 1)
        self.data, self.labels = self.load_data(kwargs.pop('class_labels', False))
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
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx,:]).contiguous().view(1,28,28)
        label = torch.from_numpy(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        
        return image.type(torch.DoubleTensor), label
    
    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    
    parser.add_argument('-model', default='cnn', type=str, help='Type of model to use')
    parser.add_argument('-ver', default=2, type=int, help='version of data to use')
    
    args = parser.parse_args()

    # Define the class labels
    class_labels = {str(x):x for x in range(10)}
    class_labels.update({'+':10, 'times':11, '-':12 })
    label_class = dict( zip(class_labels.values(), class_labels.keys() ))

    # Hyper Parameters
    num_epochs = 3
    batch_size = 20
    learning_rate = 0.001

    # MNIST Dataset
    '''train_dataset = dsets.MNIST(root='./data/',
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data/',
                            train=False, 
                            transform=transforms.ToTensor())'''

    train_dataset = EqDataLoader(class_labels=class_labels, data_ver=args.ver, train=True)

    test_dataset = EqDataLoader(class_labels=class_labels, data_ver=args.ver, train=False)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    cnn = CNN()
    cnn.double()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels).long()

            #print(images.size(), labels.size())
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            y_pred = outputs.float()
            y_true = torch.max(labels, 1)[1].long()

            #print(y_pred.size(), y_true.size())

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Test the Model
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        labels = labels.long()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    # Save the Trained Model
    torch.save(cnn, 'cnn{0}.pt'.format(args.ver))