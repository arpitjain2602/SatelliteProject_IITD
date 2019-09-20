import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix

# network for asset ,here we are aadding the edges of the network to define the network. We need to define only one network 
#for one variable. so out of these 4 networks defined in in later parts of the program only one need to be active at any time
model_asset = BayesianModel([('Formal Employment','Literacy'),
                             ('Literacy','MSL_Change'),
                             ('Literacy','Current Status'), 
                             ('Literacy','Asset_Change'), 
                             ('Formal Employment','Current Status'), 
                             ('Formal Employment','MSL_Change'), 
                             ('Formal Employment','Asset_Change'), 
                             ('MSL_Change','Asset_Change'), 
                             ('MSL_Change','Current Status'), 
                             ('Current Status','Asset_Change')                        
                           ])


# network for BF, although the name is same as model_asset ( same as previous model) but edges are different . the name is kept same 
#so that we need not to change the model name when we run the code for different varibales
model_asset = BayesianModel([('Formal Employment','Literacy'),
                             ('Literacy','MSL_Change'),
                             ('Literacy','Current Status'), 
                             ('Literacy','BF_Change'), 
                             ('Formal Employment','Current Status'), 
                             ('Formal Employment','MSL_Change'), 
                             ('Formal Employment','BF_Change'), 
                             ('MSL_Change','Current Status'), 
                             ('MSW_Change','Current Status'), 
                             ('Current Status','BF_Change')                        
                           ])


# network for FC
model_asset = BayesianModel([('Formal Employment','Literacy'),
                             ('Literacy','MSL_Change'),
                             ('Literacy','Current Status'), 
                             ('Formal Employment','Current Status'), 
                             ('Formal Employment','MSL_Change'), 
                             ('Formal Employment','FC_Change'), 
                             ('MSL_Change','Current Status'), 
                             ('Current Status','FC_Change')                        
                           ])


# Network for CHH
model_asset = BayesianModel([('Formal Employment','Literacy'),
                             ('Literacy','Current Status'), 
                             ('Formal Employment','Current Status'), 
                             ('Formal Employment','CHH_Change'), 
                             ('Current Status','CHH_Change')
                           ])

# this is just a blank file , we are reading it to get coulmn names. we will store our results in this file
df_result = pd.read_csv("D:\\Satl_project\\correct\\bayesian\\b2_input.csv")

# This is the input file which contains input data. here there is a slight change. in actual we have 3 levels level-1/2/3
# but in this file the levels are 0/1/2 because by default it starts from 0 so we have renamed the actual levels , 1->0,2->1,3->2
df = pd.read_csv("D:\\Satl_project\\correct\\bayesian\\b3_input.csv")

df_test = df.iloc[401:501,:] # for five fold cross validation we need to run this code 5 times with different range. like 0-101,101,201 and so on
a = df_test.index
df_train = df.drop(df.index[a])

model_asset.fit(df_train)
model_asset.get_cpds() 
model_asset.get_cardinality()
infer_asset = VariableElimination(model_asset)
df_test['Bayesian_label'] = 0
df_test = df_test.reset_index()
df_test = df_test.drop(['index'],axis = 1)
   
# print df_test
for index,row in df_test.iterrows():
        #print index
        a,b,c = row['Literacy'],row['Formal Employment'],row['Current Status']
        #print a,b,c,d,e
        q_asset = infer_asset.query(['CHH_Change'], evidence = {'Literacy':a,
                                                              'Formal Employment':b,
                                                              'Current Status':c}
                                                           )
        # in the above statement we are predicting CHH change
        d = q_asset ['CHH_Change']
        #print d.values[0],d.values[1]
        #print d.values[0] <= d.values[1]
        if (d.values[0] <= d.values[1]):
            df_test.iloc[index,15] = 1
df_result = pd.concat([df_result,df_test], axis=0, ignore_index=True)
df_result.to_csv('D:\\Satl_project\\correct\\bayesian\\result_BF_all_new.csv', index=True) # final results


# In[ ]:


# this code will give the accuracy results
df_test1=pd.read_csv("D:\\Satl_project\\correct\\bayesian\\result_BF_all_new.csv")
y = df_test1['CHH_Change']
predictions = df_test1['Bayesian_label']
print(accuracy_score(y,predictions))
cm = confusion_matrix(y,predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm.diagonal())
print(classification_report(y,predictions))

