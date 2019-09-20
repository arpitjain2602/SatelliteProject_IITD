import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np

df=pd.read_csv("D:\\Satl_project\\correct\\arule\\asset.csv")
frequent_itemsets = apriori(df, min_support=0.03, use_colnames=True)
frequent_itemsets.head()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#rules
rules.to_csv('D:\\Satl_project\\correct\\arule\\FC1_ARM.csv', index=True)




