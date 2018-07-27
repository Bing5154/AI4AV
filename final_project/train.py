import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from architecture import ConvNet  # from architecture.py import the class you build, fill in the blank!
from data_preprocess import train_loader #from data_preprocessing import training dataset
import random

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
learning_rate = 0.001


#Initiate the model object using the class we've already defined
num_classes = 1
#Move the model object to the Device
model = ConvNet(num_classes)
model = model.to(device)

#choose your desired optimizer##
optimizer = optim.Adam(model.parameters(), learning_rate)

# Define training loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move tensors to the configured device
        data = data.to(device)
        target = target.to(device)
        ########### Forward pass #############
         #feedforward   
        y = model(data) 
        #cross-entropy regression problems
        current_loss = F.mse_loss(y, target)
        # Backward and optimize
        model.zero_grad()
        current_loss.backward()
        optimizer.step()
 
        #######################################
        if batch_idx % 98 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                98 * batch_idx / len(train_loader), current_loss.item()))


# Train the model
for t in range(50):
  train(t)

# save the model
torch.save(model.state_dict(), 'model.ckpt')
