


import pandas as pd
import numpy as np
import pickle

df1=pd.read_csv('BF_change.csv')
df2=pd.read_csv('EMP_change.csv')
df3=pd.read_csv('FC_change.csv')
df4=pd.read_csv('MSL_change.csv')
df5=pd.read_csv('MSW_change.csv')
df6=pd.read_csv('ASSET_change.csv')


df=pd.merge(df1,df2,left_on='District_code',right_on='District_code',how='inner')

df=pd.merge(df,df3,left_on='District_code',right_on='District_code',how='inner')

df=pd.merge(df,df4,left_on='District_code',right_on='District_code',how='inner')

df=pd.merge(df,df5,left_on='District_code',right_on='District_code',how='inner')

df=pd.merge(df,df6,left_on='District_code',right_on='District_code',how='inner')


df.to_csv('future_prediction.csv')
