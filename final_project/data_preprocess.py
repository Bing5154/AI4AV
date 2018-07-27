import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import csv

#Type your own directory where you stored the data, fill in the blank
FOLDER_DATASET = "/home/ai4av/TeamOne/TeamOne18-07-26-10:32:27/"
# plt.ion()

class DriveData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        with open(folder_dataset + "steering.csv") as f:
            reader = csv.reader(f)
            for line in reader:
                # Image path
                self.__xs.append(line[2])        
                # Steering wheel label
                self.__ys.append(np.float(line[0]))

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        label = torch.from_numpy(np.asarray((self.__ys[index]-6000)/2000).reshape(1)).float()
        # random flip the images and reverse the steering command, DATA AUGMENTATION
        if random.randint(0, 1) == 0:
            img = flip(img)
            label = -label

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Convert image and label to torch tensors
            img = np.transpose(np.asarray(img),(2,0,1)) #size [120,160,3]--->[3,120,160]
            img = torch.from_numpy(img/255.0) #size [3,120,160]
        img = img[:,60:120,:]
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

# Please vist https://pytorch.org/docs/stable/torchvision/transforms.html to see different kinds of transformations
preprocessing = transforms.Compose([
   transforms.ColorJitter(brightness = 0.2),
   transforms.ToTensor(),
])


flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
])


dset_train = DriveData(FOLDER_DATASET, transform=preprocessing)
train_loader = DataLoader(dset_train, batch_size=100, shuffle= True, num_workers=1) #fill in the blank


# # Get a batch of training data, for debugging
#imgs, steering_angle = next(iter(train_loader))
#print('Batch shape:',imgs.size())
#print(steering_angle)

#plt.imshow(np.transpose(imgs.numpy()[0,:,:,:],(1,2,0)))
#plt.show()
#plt.imshow(np.transpose(imgs.numpy()[-1,:,:,:],(1,2,0)))
#plt.show()
