import pandas as pd
import numpy as np


data_old = pd.read_csv("/opt/data/AnSh/backup/master_key.csv")


row_sum = data_old.iloc[:, 12:].sum()
row_mean = data_old.iloc[:, 12:].mean()
[row_sum, row_mean]

for i in range(0,len(data_old.columns)):
    if len(pd.unique(data_old.iloc[:,i])) <= 12:
        print([data_old.columns[i],pd.unique(data_old.iloc[:,i])])

data_old.describe()
