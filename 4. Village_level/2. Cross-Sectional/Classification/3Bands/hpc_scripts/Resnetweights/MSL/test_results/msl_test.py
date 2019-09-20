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
df=pd.read_csv('/home/cse/mtech/mcs172873/Labels/VillageLabels_MSL.csv')
village_code=df["Town/Village"].values
emp_label=df["Village_HHD_Cluster_MSL"].values
actual_labels= [ int(c) for c in emp_label]
s1 = pd.Series(actual_labels,index=list(village_code))

resizeDim=224
nchannels=3
batch_size=64
numclasses=3


#------------------------------------Pre-Processing--------------------------------------------
#Importing the images

# Training Files
image_path_training = "/scratch/cse/mtech/mcs172873/data_landsat/msl/train/"
dirs1 = os.listdir(image_path_training)
train_files = []
for direc1 in dirs1:
       file1=glob.glob(os.path.join(image_path_training,direc1))
       train_files.extend(file1)
n_train = len(train_files)
print('Training set length:', n_train)

# Test files
#image_path_testing = "/scratch/cse/mtech/mcs172873/data_landsat/msl/test/"
image_path_testing = "/scratch/cse/mtech/mcs172873/data_landsat/msl/train/"
dirs2 = os.listdir(image_path_testing)
test_files = []
for direc2 in dirs2:
       file1=glob.glob(os.path.join(image_path_testing,direc2))
       test_files.extend(file1)
global n_test
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
			if(dataAll.shape[0]>resizeDim or dataAll.shape[1]>resizeDim):
				continue

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

			band2=data[:,:,1]
			band3=data[:,:,2]
			band4=data[:,:,3]
			combinedData=np.dstack((band2,band3,band4))
			left=(resizeDim-combinedData.shape[0])//2
			right=resizeDim-combinedData.shape[0]-left
			up=(resizeDim-combinedData.shape[1])//2
			down=resizeDim-combinedData.shape[1]-up

			data1=np.lib.pad(combinedData,[(left,right),(up,down),(0,0)],'constant')
			data1=np.reshape(data1,(1,resizeDim,resizeDim,nchannels))
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
		if(dataAll.shape[0]>resizeDim or dataAll.shape[1]>resizeDim):
			continue

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

		band2=data[:,:,1]
		band3=data[:,:,2]
		band4=data[:,:,3]
		combinedData=np.dstack((band2,band3,band4))

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


#----------------------Creating and Training Model-----------------------------------------------
'''
class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
		val_targ = self.model.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict)
		_val_recall = recall_score(val_targ, val_predict)
		_val_precision = precision_score(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print('-- val_f1: %f -- val_precision: %f -- val_recall %f'%(_val_f1, _val_precision, _val_recall))
		return
metrics = Metrics()
'''


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

model_weight = ['/scratch/cse/mtech/mcs172873/weights/resnet_imbalance_landat2011/msl/weights-improvement-03.hdf5', 
		   '/scratch/cse/mtech/mcs172873/weights/resnet_imbalance_landat2011/msl/weights-improvement-04.hdf5']

counter=0
for m in model_weight:
	base_model = ResNet50(weights=None, include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(3, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	print ('Loading Weights...')
	model.load_weights(m)

	print('Compiling Model...')
	model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy',metrics=[F1])

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
	path_file = '/home/cse/mtech/mcs172873/LANDSAT/Resnetweights/MSL/test_results/Train_Results_MSL-0{}.csv'.format(counter)
	np.savetxt(path_file,arr)
	counter=counter+1