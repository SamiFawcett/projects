import numpy as np
import pandas as pd
import re
import nltk
import os
import json
import io
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.spatial.distance import pdist, squareform
from subprocess import check_output
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.utils.data as dat
import torch.optim as optim
from tqdm import tqdm


# nltk.download('stopwords')


# data classes
class Database(torch.utils.data.Dataset):
    def __init__(self, path, condense):
        self.path = path
        self.condense = condense
        self.f = pd.read_csv(path)
        if(not self.condense):
            self.X, self.Y = self.create_features_labels()
        else:
            self.X, self.Y = self.condense_features()
        self.num_features = len(self.X[0])

    def condense_features(self):
        self.X, self.Y = self.create_features_labels()

        # your pc stuff here
        pca = PCA(0.99999)
        pc = pca.fit_transform(self.X)

        print(pc.shape)

        self.f_pc_x = np.array(pc)

        return self.f_pc_x, self.Y

    def check_na(self):
        count_NA = self.f.isna().sum().sum()
        print('There are %i null entries detected' % count_NA)
        if count_NA != 0:
            print('Cleaning data ----')
            self.f.dropna()
            count_NA = self.f.isna().sum().sum()
            print('There are now $i null entries' % count_NA)

    def del_id_zeros(self):
        self.X = self.X[self.X['pubchem_id'] != 0]
        self.Y = self.Y[self.X['pubchem_id'] != 0]
        return self.X, self.Y

    ''' create features and labels '''

    def create_features_labels(self):
        self.check_na()
        label = self.f.iloc[:, -1].to_numpy()  # label is our last column
        features_df = self.f.drop(['Unnamed: 0', 'pubchem_id', 'En'], axis=1)
        features = features_df.to_numpy()
        X = features_df.to_numpy()
        Y = label
        return (X, Y)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if(self.idx < len(self)):
            feature, label = self.X[self.idx], self.Y[self.idx]
            self.idx += 1
            return feature, label
        else:
            raise StopIteration

    ''' number of observations '''

    def __len__(self):
        return len(self.X)

    ''' return a tuple of attributes and label '''

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, X, Y, block_size):
        self.X = X
        self.Y = Y
        self.block_size = block_size

    '''number of blocks of size self.block_size'''

    def __len__(self):
        return len(self.X) // self.block_size

    ''' generate feature label pairs '''

    def __iter__(self):
        features = []
        labels = []
        for i in range(len(self.X)):
            feature = self.X[i]
            label = self.Y[i]
            features.append(feature)
            labels.append(label)
            if(len(features) >= self.block_size):

                yield torch.Tensor(features), torch.Tensor(labels)
                features = []
                labels = []


class IterableDatabase(torch.utils.data.IterableDataset):

    def __init__(self, path, condense, chunk_size):
        self.path = path
        self.condense = condense
        self.chunk_size = chunk_size

    def __iter__(self):
        self.chunk_idx = 0
        self.data = pd.read_csv(self.path, chunksize=self.chunk_size)
        return self

    def __next__(self):
        chunk = next(iter(self.data))
        self.chunk_idx += 1
        label = chunk.iloc[:, -1].to_numpy()  # label is our last column
        features_df = chunk.drop(['Unnamed: 0', 'pubchem_id', 'En'], axis=1)
        features = features_df.to_numpy()
        self.X = features_df.to_numpy()
        self.Y = label

        return (torch.Tensor(self.X), torch.Tensor(self.Y))

# model classes


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=1, drop_rate=0.2):
        super(FeedForwardNetwork, self).__init__()
        self.inp_dim = input_dim
        self.hid_dim = hidden_dim
        self.out_dim = out_dim

        ''' feed forward network '''
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim//2, out_dim)
        )

    def forward(self, features):
        features = self.ffn(features)
        features = torch.squeeze(features)  # [1, 1] --> [] tensor
        return features


# device
if torch.cuda.is_available():
    device = "cuda"
    print("using device", torch.cuda.get_device_name(device))
else:
    device = "cpu"

batch_size = 20  # number of batches
block_size = 1024  # number of feature label pairs in a single batch

# data
train_loader = dat.DataLoader(IterableDatabase(
    "./data/csvs/molecules_p_00000001_00225000_train_molecules.csv", False, 1024))
test_loader = dat.DataLoader(IterableDatabase(
    "./data/csvs/molecules_p_00150001_00175000.csv", False, 1024))


# model parameters
input_dim = 1289  # num of features
hidden_dim = 2048
epochs = 10
learning_rate = 0.00003
loss_fn = nn.MSELoss(reduction="mean")
net = FeedForwardNetwork(input_dim, hidden_dim)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
net.train()
net

for e in range(epochs):
    running_loss = 0
    i = 0
    for b_idx, (features, en) in tqdm(enumerate(train_loader)):
        # send data to gpu
        features = features.to(device)
        en = en.to(device)

        # zero gradients
        net.zero_grad()

        # predict
        pred = net(features.float())

        # calculate loss
        loss = loss_fn(pred, torch.squeeze(en.float()))
        loss.backward()

        # gradient descent
        optimizer.step()

        # add loss
        running_loss += np.sqrt(loss.item())

        if(i % 50 == 0):
            print("CURRENT RMSE LOSS:", running_loss / (b_idx + 1))
        i += 1
    print("EPOCH:", e+1, ":", "RMSE LOSS:", running_loss / (b_idx + 1))
    print("-----------------------------------------------------")

# testing
with torch.no_grad():
    running_loss = 0
    for b_idx, (features, en) in tqdm(enumerate(test_loader)):
        features = features.to(device)
        en = en.to(device)
        pred = net(features.float())
        loss = loss_fn(pred, torch.squeeze(en.float()))
        running_loss += np.sqrt(loss.item())

    print("***FINAL*** RMSE LOSS: ", running_loss / (b_idx+1))
    print("-----------------------------------------------------")

#### CROSS VALIDATION ####
'''
kf = KFold(n_splits=5,shuffle=True)
#for each fold
for fold, (train_index, test_index) in enumerate(kf.split(data.X)):
  print("FOLD:", fold)
  print("-----------------------------------------------------")
    #initialize data loaders
  train_loader = dat.DataLoader(IterableDataset(data.X[train_index], data.Y[train_index], block_size=block_size), batch_size=batch_size)
  test_loader = dat.DataLoader(IterableDataset(data.X[test_index], data.Y[test_index], block_size=block_size), batch_size=batch_size)

  print(len(train_loader.dataset.X), len(test_loader.dataset.X))
  #training
  for e in range(epochs):
    running_loss = 0
    for b_idx, (features, en) in tqdm(enumerate(train_loader), total=len(train_loader.dataset)//batch_size):
      #send data to gpu
      features = features.to(device)
      en = en.to(device)

      #zero gradients
      net.zero_grad()

      #predict
      pred = net(features.float())
      
      #calculate loss
      loss = loss_fn(pred, torch.squeeze(en.float()))
      loss.backward()
      
      #gradient descent
      optimizer.step()

      #add loss
      running_loss += np.sqrt(loss.item())

    print("EPOCH:", e+1, ":", "RMSE LOSS:", running_loss / (b_idx + 1))
    print("-----------------------------------------------------")

  #testing
  with torch.no_grad():
    for b_idx, (features, en) in tqdm(enumerate(test_loader), total=len(test_loader.dataset)//batch_size):
        features = features.to(device)
        en = en.to(device)
        pred = net(features.float())
        loss = loss_fn(pred, torch.squeeze(en.float()))
        running_loss += np.sqrt(loss.item())

    print("***FINAL*** RMSE LOSS: ", running_loss / (b_idx+1))
    print("-----------------------------------------------------")
'''
