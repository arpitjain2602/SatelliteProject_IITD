
# coding: utf-8

# In[24]:

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
# get_ipython().magic(u'matplotlib inline')
import plotly.offline as pyo
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.utils import shuffle
# pyo.offline.init_notebook_mode()
#setting figure size
from matplotlib.pylab import rcParams
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# In[2]:
# Read CSV

df = pd.read_csv('combined_x_y.csv')
df1 = pd.DataFrame()
x11='ASSET'
def f(row):
    if row[x11+'_2011_x'] <= row[x11+'_2001_x']:
        val = 0
    elif row[x11+'_2011_x'] > row[x11+'_2001_x']:
        val = 1
    return val
seed=7
# In[6]:
for indicator in ['ASSET','MSL','MSW','BF','FC','EMP']:


# indicator='ASSET'
# # Input Columns
	print("----------------------------------------------------------------------------")
	print("\n\n\n"+indicator)
	X_cols = [indicator+'_2001_y',indicator+'_2003',indicator+'_2005',indicator+'_2007',indicator+'_2009',indicator+'_2011_y']




	X = df[X_cols]
	# X=scale(X, axis=0)



	code = df['District']
	# x11=
	# df[indicator+'_change_x'] = df.apply(f, axis=1)
	df[indicator+'_change_x']=np.where(df[indicator+'_2011_x'] - df[indicator+'_2001_x'] > 0 , 1, 0)
	# df[indicator+'_change_x']=np.sign(df[indicator+'_2011_x'] - df[indicator+'_2001_x'])#df[indicator+'_2011_x']-df[indicator+'_2001_x']
	y = df[indicator+'_change_x']


	#count the no of values in each class
	y.value_counts()

	kfold = KFold(n_splits=5, shuffle=False, random_state=seed)

	from sklearn.ensemble import RandomForestClassifier
	print("\nRandomForests\n")	

	
	# ====================== cross validation =================
	for i in [25]:
		model = RandomForestClassifier(n_estimators=i)
		from sklearn.metrics import classification_report,accuracy_score
		print('I is:',i)
		# print(accuracy_score(y_test,predictions))
		# print(classification_report(y_test,predictions))
		results = cross_val_predict(model, X, y.ravel(), cv=kfold)
		# print(X,results)
		df1[indicator+'_change']= y
		df1[indicator] = results
		print(accuracy_score(y,results))
		print(classification_report(y,results))
		# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	


	# ======================= for saving the model==============
	# model = RandomForestClassifier(n_estimators=25)
	# model.fit(X,y.ravel())
	# filename = 'models/'+indicator+'.sav'
	# pickle.dump(model, open(filename, 'wb'))



	print("----------------------------------------------------------------------------")	

df1['District'] = df['District']
df1.to_csv('predict_new.csv',index=False)