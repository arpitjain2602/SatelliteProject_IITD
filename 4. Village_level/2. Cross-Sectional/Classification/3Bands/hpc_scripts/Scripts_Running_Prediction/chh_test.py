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


# --------------------------------------------------INSTRUCTIONS--------------------------------------------------
# VARIABLES TO BE CHANGED :
# 1. test_images_path (LINE 45 - path where test images are present)
# 2. weights_path - LINE 47, 48, 49 (Basically put the paths of all the weights in weights_path list)
# 3. pd.read_csv('/home/cse/mtech/mcs172873/Labels/VillageLabels_BF.csv') - LINE 52
# 4. Last lines (where we save the outputs) - LINE 247
# ----------------------------------------------------------------------------------------------------------------

# path variables
test_images_path = "/scratch/cse/mtech/mcs172873/All_Images_Landsat2001"

weights_path = ['/scratch/cse/mtech/mcs172873/weights/resnet_imbalance_landat2011/chh/weights-improvement-02.hdf5', 
		   '/scratch/cse/mtech/mcs172873/weights/resnet_imbalance_landat2011/chh/weights-improvement-03.hdf5', 
		   '/scratch/cse/mtech/mcs172873/weights/resnet_imbalance_landat2011/chh/weights-improvement-04.hdf5']

#Extracting Labels
df=pd.read_csv('/home/cse/mtech/mcs172873/Labels/VillageLabels_CHH.csv')
village_code=df["Town/Village"].values
emp_label=df["Village_HHD_Cluster_CHH"].values
actual_labels= [ int(c) for c in emp_label]
s1 = pd.Series(actual_labels,index=list(village_code))

resizeDim=224
nchannels=3
batch_size=64
numclasses=4


#------------------------------------Pre-Processing--------------------------------------------
#Importing the images

# Test files
image_path_testing = test_images_path
dirs2 = os.listdir(image_path_testing)
test_files = []
for files in dirs2:
    file_path = os.path.join(image_path_testing, files)
    test_files.append(file_path)
global n_test
n_test = len(test_files)
print('Test set length:', n_test)



#-------------------Creating function for Fit Generator-----------------------------------------


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

		band2=data[:,:,0]
		band3=data[:,:,1]
		band4=data[:,:,2]
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


model_weight = weights_path

counter=0
for m in model_weight:
	base_model = ResNet50(weights=None, include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(4, activation='softmax')(x)

	model = Model(input=base_model.input, output=predictions)

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
	path_file = '/home/cse/mtech/mcs172873/Test_Run/Test_Results_CHH-0{}.csv'.format(counter+2)
	np.savetxt(path_file,arr)
	counter=counter+1