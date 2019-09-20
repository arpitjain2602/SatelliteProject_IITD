from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import rasterio
import pickle
import os

vgg16 = VGG16(weights='imagenet', include_top=False)

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

village_feature = {}
i = 0
folder_path = r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\process_data\broken_files_woDupli"


for files in os.listdir(folder_path):
    
        village_code = files.split('@')[3].split('.')[0]
        filename = os.path.join(folder_path, files)

        dataset = rasterio.open(filename)
        band1 = dataset.read(1)
        band2 = dataset.read(2)
        band3 = dataset.read(3)
        cd = np.dstack((band1, band2, band3))
    #     print('Original Shape', cd.shape)

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
    #     print('New Shape', data.shape)
        cd = np.expand_dims(data, axis=0)
        cd = preprocess_input(cd)
        vgg16_feature = vgg16.predict(cd)
    #     print(vgg16_feature.shape)
        vgg16_feature = vgg16_feature[0]
    #     print(vgg16_feature.shape)

        village_feature[village_code] = vgg16_feature

pickle.dump(village_feature,open('village_feature.pkl','wb'))