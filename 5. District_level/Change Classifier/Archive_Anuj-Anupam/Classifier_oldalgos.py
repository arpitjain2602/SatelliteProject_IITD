
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

seed=7
# In[6]:
for indicator in ['ASSET','MSL','MSW','BF','FC','EMP']:

# indicator='ASSET'
# # Input Columns
	print("----------------------------------------------------------------------------")
	print("\n\n\n"+indicator)
	X_cols = [indicator+'_2001_y',indicator+'_2003',indicator+'_2005',indicator+'_2007',indicator+'_2009',indicator+'_2011_y']




	X = df[X_cols]
	X=scale(X, axis=0)



	code = df['District']
	y = df[indicator+'_2011_x']-df[indicator+'_2001_x']


	#count the no of values in each class
	y.value_counts()


	# X, X_test, y, y_test = train_test_split(X,y,test_size=0,random_state=0)
	X, y, code = shuffle(X,y,code)

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
	# In[18]:
	print("Number transactions X_train dataset: ", X_train.shape)
	print("Number transactions y_train dataset: ", y_train.shape)
	print("Number transactions X_test dataset: ", X_test.shape)
	print("Number transactions y_test dataset: ", y_test.shape)

	from sklearn.model_selection import GridSearchCV
	from sklearn.linear_model import LogisticRegression

	parameters = {
		'C': np.linspace(1, 10, 10)
				 }
	lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)

	kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

	results = cross_val_score(clf, X, y.ravel(), cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	

	lr1 = LogisticRegression(C=2.0,penalty='l2', verbose=5)
	lr1.fit(X_train, y_train.ravel())
	results = cross_val_score(lr1, X, y.ravel(), cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	# In[31]:


	# predictions = lr1.predict(X_test)
	#predictions1 = lr1.predict(X1)


	# In[32]:


	from sklearn.metrics import classification_report
	# print(accuracy_score(y_test,predictions))
	#print(accuracy_score(y1,predictions1))


	# In[35]:


	# print(classification_report(y_test,predictions))


	# **Random Forest**

	# In[41]:


	from sklearn.ensemble import RandomForestClassifier


	# In[42]:


	for i in range(10,20):
		model = RandomForestClassifier(n_estimators=i)
		model.fit(X_train,y_train)
		predictions = model.predict(X_test)
		from sklearn.metrics import classification_report,accuracy_score
		print('I is:',i)
		# print(accuracy_score(y_test,predictions))
		# print(classification_report(y_test,predictions))
		results = cross_val_score(model, X, y.ravel(), cv=kfold)
		print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	# **SVM**

	# In[84]:


	from sklearn.svm import LinearSVC


	# In[86]:


	model = LinearSVC(multi_class='ovr')
	model.fit(X_train,y_train)
	predictions = model.predict(X_test)
	from sklearn.metrics import classification_report
	# print(accuracy_score(y_test,predictions))    
	# print(classification_report(y_test,predictions))    
	results = cross_val_score(model, X, y.ravel(), cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


	# **KNN**

	# In[87]:


	from sklearn.neighbors import KNeighborsClassifier
	model = KNeighborsClassifier()
	# model.fit(X_train,y_train)
	# predictions = model.predict(X_test)
	from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	results = cross_val_score(model, X, y.ravel(), cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



	# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
	# kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
	# results = cross_val_score(estimator, dfx, dummy_y, cv=kfold)
	# # print(results)
	# print(k1)
	# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("----------------------------------------------------------------------------")	