# ======================================
# Torch intro 
# ======================================

# import 
import torch 

# create
torch.Tensor(3,2)
torch.zeros(3,2)
torch.ones(3,2)
torch.tensor([[1.,2.],[1.,2.],[1.,2.]])

# random
torch.rand(3,2)
torch.randn(3,2)
torch.randint(low=0, high=2, size=(3,2)).type(torch.FloatTensor)

# set seed 
torch.manual_seed(0)

# shape 
torch.zeros(3,2).shape

# math 
a = torch.zeros(3,2) + 1
b = torch.zeros(3,2) + 2
c = torch.zeros(2,3) + 3

a + b
a.add(b) # a.add_(b)
a - b 
a.sub(b) # a.sub_(b)
a * b 
a.mul(b) # a.mul_(b)
a / b
a.div(b) # a.div_(b)
a @ c
a.matmul(c) 

# accessing and assignments
b[:,0] = 1
b[0,:] = 1
b[1,1] = 0
b[2,1] = 0
b

# slicing
x = torch.zeros(2,2,3)
x[0,:,:]  
x[:,0,:] 
x[:,:,0] 

# transpose
x = torch.rand(2,3)
print(x.t())

# concatenation 
x = torch.ones(2,3)
y = torch.zeros(2,3)
torch.cat((x,y),dim=0)
torch.cat((x,y),dim=1)

# stacking
torch.stack((x,y), dim=0)
torch.stack((x,y), dim=1)

# sqeeze and unsqueeze
x = torch.rand(3,2,1)
x.squeeze()     # remove the 1 sized dimension
x.unsqueeze(0)  # with fake dimension

x.shape
x.squeeze().shape
x.unsqueeze(0).shape 

# view
x = torch.ones(2,2) + 1
x.view(-1)

# numpy
x.numpy()
torch.tensor(x.numpy())
torch.from_numpy(x.numpy())

# storage 
x.storage()

# cuda 
torch.cuda.is_available()
x.to('cuda')
x.to('cpu')



# ======================================
# Introduction to PyTorch
# ======================================


# ========== Libraries
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ========== Autograd:
a = torch.tensor([1.0], requires_grad=True) # when building your neural network, requires_grad=True is not required.
x = torch.tensor([2.0], requires_grad=True)
c = torch.tensor([3.0], requires_grad=True)

y = a*x**2+c

y.backward()
x.grad.data


# ========== nn Module:
import torch.nn as nn

input_units  = 10 # number of features
output_units = 1

model = nn.Sequential(
                nn.Linear(input_units, output_units),
                nn.Sigmoid()
            )

loss_func = nn.MSELoss()


# ========== Optim:
import torch.optim as optim

optimizer = optim.SGD (model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ========== Train:
x = torch.randn(20,10).to('cuda')
y = torch.randint(0,2, (20,1)).type(torch.FloatTensor).to('cuda')
model.to('cuda')

losses = []
for i in range(20):
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()
    if i%5 == 0: print(i, loss.item())

# ========== Terminology:
"""
epochs refer to the number of times that 
the entire dataset is passed forward and backward 
through the network architecture. 

batch_size refers to the number of training examples 
in a single batch (a slice of the dataset).

iterations refer to the number of batches required to complete
one epoch
"""


# ========== Plot:
import matplotlib.pyplot as plt

plt.plot(range(0,20), losses)
plt.show()


# ========== Activity:
import pandas as pd
import matplotlib.pyplot as plt


torch.manual_seed(0)

data = pd.read_csv("SomervilleHappinessSurvey2015.csv")
data.head()

x = torch.tensor(data.iloc[:,1:].values).float()
y = torch.tensor(data.iloc[:,:1].values).float()

model = nn.Sequential(nn.Linear(6, 1),nn.Sigmoid())

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for i in range(100):
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i%10 == 0:
        print(loss.item())

plt.plot(range(0,100), losses)
plt.show()


# ========== Module:

import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self, D_i, D_h, D_o):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(D_i, D_h)
        self.linear2 = torch.nn.Linear(D_h, D_o)
    def forward(self, x):
        z = F.relu(self.linear1(x))
        o = F.softmax(self.linear2(z))
        return o