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
# pyo.offline.init_notebook_mode()
#setting figure size
from matplotlib.pylab import rcParams
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import pickle
import csvdf = pd.read_csv('future_prediction.csv')
# df = pd.read_csv('combined_x_y.csv')

# for yr in [13,15,17]:
for indicator in ['ASSET','BF','EMP','FC','MSL','MSW']:
	print("----------------------------------------------------------------------------")
	print("\n\n\n"+indicator)
	
	X_cols = [indicator+'_09',indicator+'_11',indicator+'_13',indicator+'_15',indicator+'_17',indicator+'_19']
	# X_cols = [indicator+'_2001_y',indicator+'_2003',indicator+'_2005',indicator+'_2007',indicator+'_2009',indicator+'_2011_y']

	X = df[X_cols]
	# X=scale(X, axis=0)

	# Load best model
	filename = 'models/'+str(indicator)+'.sav'
	model = pickle.load(open(filename,'rb'))

	y = model.predict(X)

	with open('change/'+str(indicator)+'_change'+'.csv','w') as f:
		writer = csv.writer(f)
		writer.writerow(['District_code',str(indicator)+'_change'])
		i = 1
		for k in range(len(y)):	
			writer.writerow([i,y[k]])
			i+=1