from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam
import numpy as np
from os import walk
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy import misc
import sys
import numpy as np
import os
import pickle
from libtiff import TIFF
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()
import glob
import pandas as pd
import random
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
random.seed(2001)
print('Importing Done')


#Extracting Labels
df=pd.read_csv(r"C:\Users\AJain7\OneDrive - Stryker\Personal\Projects\Satellite Project\files for regression\Regression_Labels\BF_Actual_Label_Avg.csv")
village_code=df["Town/Village"].values
#emp_label=df["Village_HHD_Cluster_BF"].values
bf_label=df[['BF_RUD', 'BF_INT', 'BF_ADV']].values
actual_labels= [ list(c) for c in bf_label]
s1 = pd.Series(actual_labels,index=list(village_code))

resizeDim=64
nchannels=3
batch_size=64
numclasses=3

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

#------------------------------------Pre-Processing--------------------------------------------
#Importing the images

# Training Files
image_path_training = "/scratch/cse/mtech/mcs172873/data_landsat/bf/train/"
dirs1 = os.listdir(image_path_training)
train_files = []
for direc1 in dirs1:
       file1=glob.glob(os.path.join(image_path_training,direc1))
       train_files.extend(file1)
n_train = len(train_files)
print('Training set length:', n_train)

# Test files
image_path_testing = "/scratch/cse/mtech/mcs172873/data_landsat/bf/test/"
dirs2 = os.listdir(image_path_testing)
test_files = []
for direc2 in dirs2:
       file1=glob.glob(os.path.join(image_path_testing,direc2))
       test_files.extend(file1)
n_test = len(test_files)
print('Test set length:', n_test)

#-------------------Creating function for Fit Generator-----------------------------------------

# Creating function for getting the training data in batches
train_vector=np.arange(n_train)
train_vector=np.asarray(train_vector,dtype=np.int32)
random.shuffle(train_vector)

def get_batch_data():
    global train_vector,train_files,s1
    i=0
    j=0
    i=0
    k=0
    while True:
        random.shuffle(train_vector)
        X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
        Y=np.zeros((batch_size,numclasses))

        for ind in train_vector:
            tif = TIFF.open(train_files[ind], mode='r')
            image = tif.read_image()
            dataAll = np.array(image)

            village_code=int(((train_files[ind].split('@')[3]).split('.')[0]).split()[0])
            val=0
            try:
                try:
                    val=int(s1.loc[village_code])-1 # why did this??
                except:
                    continue
            except:
                continue


            # Commented because I dont have all the bands    
            data=np.delete(dataAll,[11,12],axis=2)
            band2=data[:,:,0]
            band3=data[:,:,1]
            band4=data[:,:,2]
            cd=np.dstack((band2,band3,band4))
            #print('shape of cd',cd.shape)


            if (cd.shape[0] > resizeDim or cd.shape[1] > resizeDim):
                #print('shape before slicing',cd.shape)
                #print('slicing x')
                if(cd.shape[0] > resizeDim and cd.shape[1] > resizeDim):
                    #print('Slicing both together')
                    combinedData = slice_x(cd,resize_dim_x=resizeDim)
                    combinedData = slice_y(combinedData, resize_dim_y=resizeDim)
                    #print('shape after slicing',combinedData.shape)
                    
                elif(cd.shape[0] > resizeDim and cd.shape[1] <= resizeDim):
                    #print('Slicing x')
                    combinedData = slice_x(cd,resize_dim_x=resizeDim)
                    #print('shape after slicing',combinedData.shape)
                elif(cd.shape[0] <= resizeDim and cd.shape[1] > resizeDim):
                    #print('slicing y')
                    combinedData = slice_y(cd, resize_dim_y=resizeDim)
                    #print('shape after slicing',combinedData.shape)
            else:
                combinedData = cd
                #print('Shape without slicing', combinedData.shape)

            left=(resizeDim-combinedData.shape[0])//2
            right=resizeDim-combinedData.shape[0]-left

            #print('Left Right',left, right)
            up=(resizeDim-combinedData.shape[1])//2
            down=resizeDim-combinedData.shape[1]-up

            #print('UP, Down',up, down)
            data1=np.lib.pad(combinedData,[(left,right),(up,down),(0,0)],'constant')
            data1=np.reshape(data1,(1,resizeDim,resizeDim,nchannels))
            #print('shape of data1',data1.shape)
            #print('----------------------------------------------------')

            if np.isnan(data1).any():
                continue
            else:
                X=np.vstack((X,data1))
                Y[i%batch_size,val]=1

            i+=1

            if i%(64)==0:
                X=np.asarray(X,dtype=np.float32)
                Y=np.asarray(Y,dtype=np.int32)
                dataset = (X, Y)
                yield X,Y
                break

j=0
ind=0

# Creating function for getting the test data in batches
test_vector=np.arange(n_test)
test_vector=np.asarray(test_vector,dtype=np.int32)
random.shuffle(test_vector)

def get_eval_data():
	global j
	global ind
	global test_vector,test_files,s1

	X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
	Y=np.zeros((batch_size,numclasses))
	village_label=np.zeros((batch_size,numclasses))

	while ind< len(test_vector):

		ind=(ind+1)%len(test_vector)
		tif = TIFF.open(test_files[test_vector[ind]], mode='r')
		image = tif.read_image()
		dataAll = np.array(image)

		village_code=int( ((test_files[test_vector[ind]].split('@')[3]).split('.')[0]).split()[0] )
		val=0
		try:
			try:
				val=int(s1.loc[village_code])-1
			except:
				continue
		except:
			continue

		data=np.delete(dataAll,[11,12],axis=2)

		band2=data[:,:,0]
		band3=data[:,:,1]
		band4=data[:,:,2]
		cd=np.dstack((band2,band3,band4))

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

		left=(resizeDim-combinedData.shape[0])//2
		right=resizeDim-combinedData.shape[0]-left
		up=(resizeDim-combinedData.shape[1])//2
		down=resizeDim-combinedData.shape[1]-up

		data1=np.lib.pad(combinedData,[(left,right),(up,down),(0,0)],'constant')
		data1=np.reshape(data1,(1,resizeDim, resizeDim,nchannels))

		if np.isnan(data1).any():
			continue
		else:
			X=np.vstack((X,data1))
			Y[j%batch_size,val]=1
			village_label[j%batch_size,val]=village_code

		j+=1
		if j%(64)==0:
			X=np.asarray(X,dtype=np.float32)
			Y=np.asarray(Y,dtype=np.int32)
			village_label = np.asarray(village_label,dtype=np.int32)

			dataset = (X, Y)
			return X,Y, village_label
		
k=0
ind=0


filepath = "/scratch/cse/mtech/mcs172873/weights/resnet_imbalance_64x64/bf/weights-improvement-{epoch:02d}.hdf5"
#filepath = "/home/cse/mtech/mcs172873/LANDSAT/Resnetweights/BF/checkpoints/weights-improvement-{epoch:02d}.hdf5"
#filepath = "/home/cse/mtech/mcs172873/LANDSAT/ResNet50/BF/checkpoints/weights-improvement-04.hdf5"
checkpoint1 = ModelCheckpoint(filepath, verbose=1)
checkpoint2 = EarlyStopping(monitor='val_loss',patience=1,verbose=1,mode='auto',min_delta=0.1)
callbacks_list = [checkpoint1]

def F1(y_true, y_pred):
    def recall(y_true, y_pred):
        #Recall metric: Only computes a batch-wise average of recall.
        #Computes the recall, a metric for multi-label classification of how many relevant items are selected
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        #Precision metric: Only computes a batch-wise average of precision.
        #Computes the precision, a metric for multi-label classification of how many selected items are relevant.
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

class_weight = {0: 0.40896103511576953, 1: 3.940446473147422, 2: 3.3222489476849066}

print ('################################################################################################')
print ('Starting Transfer Learning............................')
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=[F1])
model.fit_generator(get_batch_data(), epochs=5, steps_per_epoch=int(np.ceil(n_train/float(batch_size))), verbose=1, class_weight=class_weight)
print ('Transfer Learning Finished............................')
print ('################################################################################################')

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
for layer in model.layers[:]:
   layer.trainable = True


# Load model weight since training stopped in between
#model.load_weights('/home/cse/mtech/mcs172873/LANDSAT/ResNet50/BF/checkpoints/weights-improvement-02.hdf5')

print ('################################################################################################')
print ('Starting Fine Tuning...............................')
from keras.optimizers import SGD
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics=[F1])
#model.fit_generator(get_batch_data(),epochs=5, callbacks= callbacks_list,steps_per_epoch=int(np.ceil(n_train/float(batch_size))), verbose=1)
model.fit_generator(get_batch_data(),epochs=5, class_weight=class_weight,callbacks= callbacks_list,steps_per_epoch=int(np.ceil(n_train/float(batch_size))), verbose=1)
print ('Fine Tuning Finished...............................')
print ('################################################################################################')

print ('........Training Complete..........')



print ('Starting Testing...................')
# This test is on test_data and on the model that is trained on 5 epochs.
test_batch_size=64
total_test_batch= int(n_test/64)-1
avg=0
predicted_y=np.array([])
actual_y=np.array([])
actual_village = np.array([])

for i in range(total_test_batch):
	evalX,evalY,eval_village_code=get_eval_data()
	
	loss,f1 = model.evaluate(evalX,evalY, batch_size=64)
	
	y1=model.predict(evalX,batch_size=64)
	
	y1=np.argmax(y1,axis=1)
	y2=np.argmax(evalY,axis=1)
	y3 = np.amax(eval_village_code,axis=1)
	predicted_y=np.hstack((predicted_y,y1))
	actual_y=np.hstack((actual_y,y2))
	actual_village = np.hstack((actual_village,y3))

	print("loss: ",loss)
	print("F1 Score: ",f1)
	avg+=f1
print("avg: ", avg/total_test_batch)
arr=np.vstack((actual_y,predicted_y,actual_village)).T
np.savetxt('/home/cse/mtech/mcs172873/64x64/Resnet_BF/Test_Results_BF.csv',arr)