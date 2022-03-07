print("Pytorch Tutorial")
# ========================================== 
# 1. How to Install PyTorch :
# ==========================================

# ---------------------- 
# Install PyTorch :
# ----------------------
# !pip install torch
# !pip install torchvision

# ----------------------
# Check pytorch version:
# ----------------------
# import torch
# print(torch.__version__)


# ========================================== 
# 2. PyTorch Deep Learning Model Life-Cycle:
# ==========================================

# ---------------------- 
# The five steps in the life-cycle are as follows :
# ----------------------
# 1. Prepare the Data.
# 2. Define the Model.
# 3. Train the Model.
# 4. Evaluate the Model.
# 5. Make Predictions.

# ----------------------
#  Step 1: Prepare the Data
# ----------------------
# PyTorch provides the Dataset class that you can extend and customize to load your dataset.
# For example: 
# The constructor of your dataset object can load your data file (e.g. a CSV file). 
# You can then override: 
# __len__() function that can be used to get the length of the dataset (number of rows or samples), 
# __getitem__() function that is used to get a specific sample by index.
"""
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        self.X = ...
        self.y = ...

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
"""

# A DataLoader instance can be created for the training dataset, test dataset, and even a validation dataset.
# The random_split() function can be used to split a dataset into train and test sets. 
# Once split, a selection of rows from the Dataset can be provided to a DataLoader, 
# along with the batch size and whether the data should be shuffled every epoch.
"""
# create the dataset
dataset = CSVDataset(...)

# select rows from the dataset
train, test = random_split(dataset, [[...], [...]])

# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)

# train the model
for i, (inputs, targets) in enumerate(train_dl):
"""

# ---------------------- 
# Step 2: Define the Model
# ----------------------
# The constructor of your class defines the layers of the model 
# and the forward() function is the override that defines how to forward propagate input through the defined layers of the model.
"""
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 1)
        self.activation = Sigmoid()

    # forward propagate input
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
"""

# The weights of a given layer can also be initialized after the layer is defined in the constructor.
"""
xavier_uniform_(self.layer.weight)
"""

# ---------------------- 
# Step 3: Train the Model
# ----------------------
# The training process requires that you define 
# a loss function and an optimization algorithm.
"""
# define the optimization
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
"""

# Training the model involves enumerating the DataLoader for the training dataset.
# First, a loop is required for the number of training epochs. 
# Then an inner loop is required for the mini-batches for stochastic gradient descent.
"""
# enumerate epochs
for epoch in range(100):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
    	...
"""

# Each update to the model involves the same general pattern comprised of:
#   - Clearing the last error gradient.
#   - A forward pass of the input through the model.
#   - Calculating the loss for the model output.
#   - Backpropagating the error through the model.
#   - Update the model in an effort to reduce loss.
# For example: 
"""
# clear the gradients
optimizer.zero_grad()
# compute the model output
yhat = model(inputs)
# calculate loss
# loss = criterion(yhat, targets)
# credit assignment
loss.backward()
# update model weights
optimizer.step()
"""

# ---------------------- 
# Step 4: Evaluate the model
# ----------------------
# This can be achieved by using the DataLoader for the test dataset 
# and collecting the predictions for the test set, 
# then comparing the predictions to the expected values of the test set 
# and calculating a performance metric. 
"""
for i, (inputs, targets) in enumerate(test_dl):
    # evaluate the model on the test set
    yhat = model(inputs)
    ...
"""

# ---------------------- 
# Step 5: Make predictions
# ----------------------
# This requires that you wrap the data in a PyTorch Tensor data structure.
"""
# convert row to data
row = Variable(Tensor([row]).float())
# make prediction
yhat = model(row)
# retrieve numpy array
yhat = yhat.detach().numpy()
"""



# ========================================== 
# 3. How to Develop PyTorch Deep Learning Models:
# ==========================================

# ----------------------
# MLP for Binary Classification
# ----------------------

# dataset description:
""" https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names """


# Notes: 
"""
> This dataset involves predicting whether there is a structure in the atmosphere or not given radar returns.
> it is a good practice to use ‘relu‘ activation with a ‘He Uniform‘ weight initialization. This combination goes a long way to overcome the problem of vanishing gradients when training deep neural network models.
> The model predicts the probability of class 1 and uses the sigmoid activation function. 
> The model is optimized using stochastic gradient descent and seeks to minimize the binary cross-entropy loss.
> A Multilayer Perceptron model, or MLP for short, is a standard fully connected neural network model.
> An MLP is a model with one or more fully connected layers. This model is appropriate for tabular data, that is data as it looks in a table or spreadsheet 
"""

# load libraries 
""" General """
from numpy import vstack                        # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
from pandas import read_csv                     # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

""" Sklearn """
from sklearn.preprocessing import LabelEncoder  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.metrics import accuracy_score      # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

""" Torch: Data preparation """
from torch import Tensor                        # https://pytorch.org/docs/stable/tensors.html
from torch.utils.data import Dataset            # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
from torch.utils.data import DataLoader         # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
from torch.utils.data import random_split       # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

""" Torch: Model """
from torch.nn import Module                     # https://pytorch.org/docs/stable/generated/torch.nn.Module.html 
from torch.nn import Linear                     # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
from torch.nn import ReLU                       # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
from torch.nn import Sigmoid                    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
from torch.nn.init import kaiming_uniform_      # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
from torch.nn.init import xavier_uniform_       # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_

""" Torch: Loss and Optim """
from torch.nn import BCELoss                    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
from torch.optim import SGD                     # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html



# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for _ in range(100):
        # enumerate mini batches
        for _ , (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for _ , (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# Runner
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(34)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))

# ----------------------
# MLP for Multiclass Classification
# ----------------------
"""
https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/#:~:text=3.2.%20how%20to%20develop%20an%20mlp%20for%20multiclass%20classification
"""

# ----------------------
# MLP for Regression
# ----------------------
"""
https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/#:~:text=3.3.%20how%20to%20develop%20an%20mlp%20for%20regression
"""

# ----------------------
# CNN for Image Classification
# ----------------------
""" 
https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/#:~:text=3.4.%20how%20to%20develop%20a%20cnn%20for%20image%20classification
"""

