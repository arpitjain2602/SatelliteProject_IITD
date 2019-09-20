
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

df = pd.read_csv('combined_x_y_subset.csv')

seed=7
# In[6]:
for indicator in ['ASSET','MSL','MSW','BF','FC','EMP']:

# # Input Columns
	print("----------------------------------------------------------------------------")
	print("\n\n\n"+indicator)
	X_cols = [indicator+'_2001_y',indicator+'_2003',indicator+'_2005',indicator+'_2007',indicator+'_2009',indicator+'_2011_y']




	X = df[X_cols]
	# X=scale(X, axis=0)



	code = df['District']
	y = np.sign(df[indicator+'_2011_x'] - df[indicator+'_2001_x'])#df[indicator+'_2011_x']-df[indicator+'_2001_x']

	print("count the no of values in each class")
	print(y.value_counts())

	from sklearn.model_selection import GridSearchCV
	from sklearn.linear_model import LogisticRegression

	parameters = {
		'C': np.linspace(1, 10, 10)
				 }
	lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	clf = GridSearchCV(lr, parameters, cv=5, verbose=0, n_jobs=3)

	kfold = KFold(n_splits=5, shuffle=False)

	# results = cross_val_score(clf, X, y.ravel(), cv=kfold)
	# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("LogisticRegressionCLF")
	results = cross_val_predict(clf, X, y.ravel(), cv=kfold)
	print(accuracy_score(y,results))
	print(classification_report(y,results))


	lr1 = LogisticRegression(C=2.0,penalty='l2', verbose=0)
	
	# results = cross_val_score(lr1, X, y.ravel(), cv=kfold)
	# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("LogisticRegression")
	results = cross_val_predict(clf, X, y.ravel(), cv=kfold)
	print(accuracy_score(y,results))
	print(classification_report(y,results))


	from sklearn.metrics import classification_report

	# **Random Forest**
	print("RandomForestClassifier")
	from sklearn.ensemble import RandomForestClassifier
	for i in range(10,20):
		
		model = RandomForestClassifier(n_estimators=i)
		print('I is:',i)

		# results = cross_val_score(model, X, y.ravel(), cv=kfold)
		# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
		
		results = cross_val_predict(clf, X, y.ravel(), cv=kfold)
		print(accuracy_score(y,results))
		print(classification_report(y,results))


	# **SVM**
	from sklearn.svm import LinearSVC
	model = LinearSVC(multi_class='ovr')
	# results = cross_val_score(model, X, y.ravel(), cv=kfold)
	# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("LinearSVC")
	results = cross_val_predict(clf, X, y.ravel(), cv=kfold)
	print(accuracy_score(y,results))
	print(classification_report(y,results))


	# **KNN**

	# In[87]:


	from sklearn.neighbors import KNeighborsClassifier
	model = KNeighborsClassifier()
	
	# results = cross_val_score(model, X, y.ravel(), cv=kfold)
	# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("KNeighborsClassifier")
	results = cross_val_predict(clf, X, y.ravel(), cv=kfold)
	print(accuracy_score(y,results))
	print(classification_report(y,results))
	print("----------------------------------------------------------------------------")