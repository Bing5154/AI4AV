import torch 
import torch.nn as nn
import torch.nn.functional as F

#Create the model class
#Build the network

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=7, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(6, 12, kernel_size=7, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
                nn.Conv2d(12, 24, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),                
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
                nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),                
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
                nn.Conv2d(48, 96, kernel_size=2, stride=1,padding = 0),
                nn.ReLU())
        self.fc = nn.Linear(2*8*96, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out







