


import pandas as pd
import numpy as np
import pickle

df1=pd.read_csv('cloud.csv')
df2=pd.read_csv('correct_prediction.csv')


df=pd.merge(df2,df1,left_on='District',right_on='District',how='inner')


df.to_csv('corr.csv')
