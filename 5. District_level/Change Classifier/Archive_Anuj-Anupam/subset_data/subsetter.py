import pandas as pd
import numpy as np

df1=pd.read_csv('combined_x_y.csv')
df2=pd.read_csv('District_list.csv')

df=pd.merge(df1,df2,left_on='District',right_on='District',how='inner')

df.to_csv('combined_x_y_subset.csv',index=False)