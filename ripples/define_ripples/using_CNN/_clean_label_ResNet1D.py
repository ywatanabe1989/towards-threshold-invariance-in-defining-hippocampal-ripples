#!/usr/bin/env python

# from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
# # Make sure python version is compatible with pyTorch
# from cleanlab.util import VersionWarning
# python_version = VersionWarning(
#     warning_str = "pyTorch supports Python version 2.7, 3.5, 3.6, 3.7.",
#     list_of_compatible_versions = [2.7, 3.5, 3.6],
# )
# if python_version.is_compatible(): # pragma: no cover
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator


def calc_out_len(i, k, s, p, d=1):
  o = (i + 2*p - k - (k-1)*(d-1))/s + 1
  return o


def pad_len_for_the_same_length(i, k, s, d=1):
  p = ((i-1)*s -i + k + (k-1)*(d-1)) / 2
  return p


def conv_k(in_chs, out_chs, k=1, s=1, p=1):
    """ Build size k kernel's convolution layer with padding"""
    return nn.Conv1d(in_chs, out_chs, kernel_size=k, stride=s, padding=p, bias=False)


class BasicBlock(nn.Module):
    """ Basic Block using kernel sizes = (7,5,3) convolustion with padding"""
    expansion = 1

    def __init__(self, in_chs, out_chs, use_activation):
        super(BasicBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        if use_activation == 'lrelu':
          activation = nn.LeakyReLU(0.1)
        if use_activation == 'relu':
          activation = nn.ReLU()
        if use_activation == 'selu':
          activation = nn.SELU()

        self.conv7 = conv_k(in_chs, out_chs, k=7, p=3)
        self.bn7 = nn.BatchNorm1d(out_chs)
        self.activation7 = activation        # self.selu7 = nn.SELU(inplace=True) # self.relu7 = nn.ReLU()

        self.conv5 = conv_k(out_chs, out_chs, k=5, p=2)
        self.bn5 = nn.BatchNorm1d(out_chs)
        self.activation5 = activation        # self.selu5 = nn.SELU(inplace=True) # self.relu5 = nn.ReLU()

        self.conv3 = conv_k(out_chs, out_chs, k=3, p=1)
        self.bn3 = nn.BatchNorm1d(out_chs)
        self.activation3 = activation        # self.selu3 = nn.SELU(inplace=True) # self.relu3 = nn.ReLU()

        self.expansion_conv = conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = activation        # self.selu = nn.SELU(inplace=True) # self.relu = nn.ReLU()


    def forward(self, x):
        residual = x

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.activation7(x)        # x = self.selu7(x) # self.relu7(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation5(x)        # x = self.selu5(x) # self.relu5(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)        # x = self.selu3(x) # self.relu3(x)

        if self.in_chs != self.out_chs:
          residual = self.expansion_conv(residual)
        residual = self.bn(residual)

        x += residual
        x = self.activation(x)        # x = self.selu(x) # self.relu(x)

        return x


class ResNet1D(nn.Module):
    def __init__(self, ):
        super(ResNet1D, self).__init__()

        self.block1 = BasicBlock(1, 64, 'lrelu')
        self.block2 = BasicBlock(64, 128, 'lrelu')
        self.block3 = BasicBlock(128, 128, 'lrelu')

        self.gap = torch.mean

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.transpose(1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        dtype_orig = x.dtype
        x = x.to(torch.float64)
        x = self.gap(x, dim=-1)
        x = x.to(dtype_orig)

        x = self.activation(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=-1)

        return x


class CNN(BaseEstimator): # Inherits sklearn classifier
    '''Wraps a PyTorch CNN for the MNIST dataset within an sklearn template by defining
    .fit(), .predict(), and .predict_proba() functions. This template enables the PyTorch
    CNN to flexibly be used within the sklearn architecture -- meaning it can be passed into
    functions like cross_val_predict as if it were an sklearn model. The cleanlab library
    requires that all models adhere to this basic sklearn template and thus, this class allows
    a PyTorch CNN to be used in for learning with noisy labels among other things.'''
    def __init__(
        self,
        batch_size = 64,
        epochs = 6,
        log_interval = 50, # Set to None to not print
        lr = 0.01,
        momentum = 0.5,
        no_cuda = False,
        seed = 1,
        test_batch_size = MNIST_TEST_SIZE,
        # Set to 'test' to force fit() and predict_proba() on test_set
        # Be careful setting this, it will override every other loader
        # If you set this to 'test', but call .predict(loader = 'train')
        # then .predict() will still predict on test!
        loader = None,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.test_batch_size = test_batch_size

        self.cuda = not self.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.seed)
        if self.cuda: # pragma: no cover
            torch.cuda.manual_seed(self.seed)

        # Instantiate PyTorch model
        self.model = Net()
        if self.cuda: # pragma: no cover
            self.model.cuda()

        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.loader = loader


    def fit(self, train_idx, train_labels = None, loader = 'train'):
        '''This function adheres to sklearn's "fit(X, y)" format for compatibility with scikit-learn.
        ** All inputs should be numpy arrays, not pyTorch Tensors
        train_idx is not X, but instead a list of indices for X (and y if train_labels is None).
        This function is a member of the cnn class which will handle creation of X, y from
        the train_idx via the train_loader.'''

        if self.loader is not None:
            loader = self.loader
        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError("Check that train_idx and train_labels are the same length.")

        class_weight = None

        train_dataset = datasets.MNIST(
            root = '../data',
            train = (loader=='train'),
            download = True,
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # Use provided labels if not None, o.w. use MNIST dataset training labels
        if train_labels is not None:
            # Create sparse tensor of train_labels with (-1)s for labels not in train_idx.
            # We avoid train_data[idx] because train_data may very large, i.e. image_net
            sparse_labels = np.zeros(MNIST_TRAIN_SIZE if loader == 'train' else MNIST_TEST_SIZE, dtype=int) - 1
            sparse_labels[train_idx] = train_labels
            train_dataset.targets = sparse_labels

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
#             sampler=SubsetRandomSampler(train_idx if train_idx is not None else range(MNIST_TRAIN_SIZE)),
            sampler=SubsetRandomSampler(train_idx),
            batch_size=self.batch_size,
            **self.loader_kwargs
        )

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Train for self.epochs epochs
        for epoch in range(1, self.epochs + 1):

            # Enable dropout and batch norm layers
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda: # pragma: no cover
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target, class_weight)
                loss.backward()
                optimizer.step()
                if self.log_interval is not None and batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_idx),
                        100. * batch_idx / len(train_loader), loss.item()))

    def predict(self, idx = None, loader = None):
        # get the index of the max probability
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)


    def predict_proba(self, idx = None, loader = None):
        if self.loader is not None:
            loader = self.loader
        if loader is None:
            is_test_idx = idx is not None and len(idx) == MNIST_TEST_SIZE and np.all(np.array(idx) == np.arange(MNIST_TEST_SIZE))
            loader = 'test' if is_test_idx else 'train'
        dataset = datasets.MNIST(
            root = '../data',
            train = (loader=='train'),
            download = True,
            transform = transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
            )
        )
        # Filter by idx
        if idx is not None:
            if (loader == 'train' and len(idx) != MNIST_TRAIN_SIZE) or (
                loader == 'test' and len(idx) != MNIST_TEST_SIZE):
                dataset.data = dataset.data[idx]
                dataset.targets = dataset.targets[idx]

        loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size=self.batch_size if loader=='train' else self.test_batch_size,
            **self.loader_kwargs
        )

        # sets model.train(False) inactivating dropout and batch-norm layers
        self.model.eval()

        # Run forward pass on model to compute outputs
        outputs = []
        for data, _ in loader:
            if self.cuda: # pragma: no cover
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                output = self.model(data)
            outputs.append(output)

        # Outputs are log_softmax (log probabilities)
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        out = outputs.cpu().numpy() if self.cuda else outputs.numpy()
        pred = np.exp(out)
        return pred




if __name__ == "__main__":
    bs = 3
    seq_len = 2000
    n_features = 1

    data = torch.randn(bs, seq_len, n_features).cuda()
    model = ResNet1D()
    model = model.cuda()

    output = model(data) # input: [batch, seq_len, n_features]
    print(output.shape)
