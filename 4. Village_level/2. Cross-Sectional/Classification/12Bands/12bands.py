import numpy as np
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import torch
from torch.autograd.variable import Variable
from torchvision import datasets, models, transforms
import math
import torch.utils.model_zoo as model_zoo
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import rasterio

# loading ResNet Model
model = models.resnet18(pretrained=True)

# Changing last layer
num_final_in = model.fc.in_features

# The final layer of the model is model.fc so we can basically just overwrite it 
# to have the output = number of classes we need. Say, 300 classes.
NUM_CLASSES = 3
model.fc = nn.Linear(num_final_in, NUM_CLASSES)

# Get old weights
old_conv_weight = model.conv1.weight.data
#print(type(old_conv_weight))

# create new conv layer (10 layer and not 13)
# Landsat 7 -> 8 bands
# Landsat 8 -> 9 bands
new_conv = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Xavier init
nn.init.xavier_normal_(new_conv.weight)

# copy old weights into first 3 channels
new_conv.weight.data[:,:3].copy_(old_conv_weight)

# replace old conv with the new one
model.conv1 = new_conv

counter = 0
for child in model.children():
    #print(child)
    print("--", counter,'----------', child)
    counter += 1


# Creating custom Dataset classes
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)    
        return x, y
    
    def __len__(self):
        return len(self.data)

#Extracting Labels
df=pd.read_csv(r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\Village_Labels.csv")  # --> Contain village label
village_code=df["Town/Village"].values
emp_label=df["Village_HHD_Cluster_MSW"].values
actual_labels= [ int(c) for c in emp_label]
s1 = pd.Series(actual_labels,index=list(village_code))

def slice_x(img,resize_dim_x):
    x,y,_ = img.shape
    startx = x//2-(resize_dim_x//2)
    return img[startx:startx+resize_dim_x, :,:]
def slice_y(img,resize_dim_y):
    x,y,_ = img.shape
    starty = y//2-(resize_dim_y//2)
    return img[:,starty:starty+resize_dim_y,:]
def crop_center(img,cropx,cropy): #Function for cropping the image from the center
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

resizeDim = 224
nchannels = 9



#--------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------Training Dataset------------------------------------------------------
all_img = []
all_label = np.array([])
train_path = r"C:\Users\AJain7\Desktop\landsat_sample_data\landsat_sample_data"  # --> Contains Satellite images
for file in os.listdir(train_path):
    filename = os.train_path.join(train_path,file)
    dataset = rasterio.open(filename)
    village_code = int(file.split('@')[3].split('.')[0])
    label = s1.loc[village_code]
    all_label = np.append(all_label,label)
    #X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
    band1 = dataset.read(1)
    band2 = dataset.read(2)
    band3 = dataset.read(3)
    band4 = dataset.read(4)
    band5 = dataset.read(5)
    band6 = dataset.read(6)
    band7 = dataset.read(7)
    band8 = dataset.read(8)
    band9 = dataset.read(9)
    band10 = dataset.read(10)
    band11 = dataset.read(11)
    band12 = dataset.read(12)
    band13 = dataset.read(13)
    
    #cd = np.dstack((band1, band2, band3, band4, band5, band6, band7, band8, band9, band10, band11, band12, band13))
    cd = np.dstack((band1, band2, band3, band4, band5, band6, band7, band8, band9))
    #cd = np.dstack((band1, band2, band3, band4, band5, band6, band7, band8)) # Change Number of Channels Variables
    
    #print(cd.shape)
    
    if (cd.shape[0] > resizeDim or cd.shape[1] > resizeDim):
        if(cd.shape[0] > resizeDim and cd.shape[1] > resizeDim):
            combinedData = slice_x(cd,resize_dim_x=resizeDim)
            combinedData = slice_y(combinedData, resize_dim_y=resizeDim)
        elif(cd.shape[0] > resizeDim and cd.shape[1] <= resizeDim):
            combinedData = slice_x(cd,resize_dim_x=resizeDim)
        elif(cd.shape[0] <= resizeDim and cd.shape[1] > resizeDim):
            combinedData = slice_y(cd, resize_dim_y=resizeDim)
    else:
        combinedData = cd
    
    left = (resizeDim-combinedData.shape[0])//2
    right = resizeDim-combinedData.shape[0] - left
    up = (resizeDim-combinedData.shape[1])//2
    down = resizeDim-combinedData.shape[1] - up
    
    data = np.lib.pad(combinedData, [(left,right),(up,down),(0,0)], 'constant')
    data = np.reshape(data,(1,nchannels,resizeDim,resizeDim))
    all_img.append(data)

ai = np.vstack(all_img)
# ai --> All images of numpy array
# all_label --> Corresponding labels
train_dataset = MyDataset(ai, all_label)
print('Number of Train Samples:',train_dataset.__len__())
print(train_dataset.__getitem__(9)[1])
#loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available() )
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------Test Dataset-------------------------------------------------------------------
all_test_img = []
all_test_label = np.array([])

test_path =   # --> Contains Satellite images
for file in os.listdir(test_path):
    filename = os.test_path.join(test_path,file)
    dataset = rasterio.open(filename)
    village_code = int(file.split('@')[3].split('.')[0])
    label = s1.loc[village_code]
    all_test_label = np.append(all_test_label,label)
    #X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
    band1 = dataset.read(1)
    band2 = dataset.read(2)
    band3 = dataset.read(3)
    band4 = dataset.read(4)
    band5 = dataset.read(5)
    band6 = dataset.read(6)
    band7 = dataset.read(7)
    band8 = dataset.read(8)
    band9 = dataset.read(9)
    band10 = dataset.read(10)
    band11 = dataset.read(11)
    band12 = dataset.read(12)
    band13 = dataset.read(13)
    
    #cd = np.dstack((band1, band2, band3, band4, band5, band6, band7, band8, band9, band10, band11, band12, band13))
    cd = np.dstack((band1, band2, band3, band4, band5, band6, band7, band8, band9))
    #cd = np.dstack((band1, band2, band3, band4, band5, band6, band7, band8)) # Change Number of Channels Variables
    
    #print(cd.shape)
    
    if (cd.shape[0] > resizeDim or cd.shape[1] > resizeDim):
        if(cd.shape[0] > resizeDim and cd.shape[1] > resizeDim):
            combinedData = slice_x(cd,resize_dim_x=resizeDim)
            combinedData = slice_y(combinedData, resize_dim_y=resizeDim)
        elif(cd.shape[0] > resizeDim and cd.shape[1] <= resizeDim):
            combinedData = slice_x(cd,resize_dim_x=resizeDim)
        elif(cd.shape[0] <= resizeDim and cd.shape[1] > resizeDim):
            combinedData = slice_y(cd, resize_dim_y=resizeDim)
    else:
        combinedData = cd
    
    left = (resizeDim-combinedData.shape[0])//2
    right = resizeDim-combinedData.shape[0] - left
    up = (resizeDim-combinedData.shape[1])//2
    down = resizeDim-combinedData.shape[1] - up
    
    data = np.lib.pad(combinedData, [(left,right),(up,down),(0,0)], 'constant')
    data = np.reshape(data,(1,nchannels,resizeDim,resizeDim))
    all_test_img.append(data)

ai = np.vstack(all_test_img)
# ai --> All images of numpy array
# all_label --> Corresponding labels
test_dataset = MyDataset(ai, all_test_label)
print('Number of Test Samples:',test_dataset.__len__())
print(test_dataset.__getitem__(9)[1])
#loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available() )
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            
            print('Epoch [{}/{}],  Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 100, loss.item()))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss Function and Optimizer
weights = [0.40896103511576953, 3.940446473147422, 3.3222489476849066]
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
els = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer_ft, els, num_epochs=5)