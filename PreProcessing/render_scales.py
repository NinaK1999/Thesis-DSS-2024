import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split

random_state = 42

Data1 = pd.read_excel('./All_data/dataset_study1_OSF.xlsx')
Data2 = pd.read_excel('./All_data/dataset_study2_OSF.xlsx')
print(Data1.shape)
target_list = ['DASS_stress', 'DASS_anxiety', "DASS_depression"]

subscales_dass = [col for col in Data1.columns if bool(re.match(r'^DASS\d+$', col))]

Data1target = Data1[target_list]
Data1 = Data1.drop(target_list + subscales_dass, axis=1)
Data1['FBI7'], Data1['FBI8'] = np.log(Data1['FBI7'] + 1), np.log(Data1['FBI8'] + 1)

Data2 = Data2.loc[Data2['FBI7'] != Data2['FBI7'].max()].loc[Data2['FBI8'] != Data2['FBI8'].max()]
Data2['FBI7'], Data2['FBI8'] = np.log(Data2['FBI7'] + 1), np.log(Data2['FBI8'] + 1)
Data2target = Data2[target_list]
Data2 = Data2.drop([*target_list, *subscales_dass, 'CI', 'CI2'], axis=1)

# Use nonparanormal normalization from 'huge' library from Rstudio
pandas2ri.activate()
robjects.r('library("huge")')

Data1_norm = pd.DataFrame(robjects.r['huge.npn'](Data1), columns=Data1.columns)
Data2_norm = pd.DataFrame(robjects.r['huge.npn'](Data2), columns=Data2.columns)

X_train, y_train = Data2_norm, Data2target
X_val, X_test, y_val, y_test = train_test_split(Data1_norm, Data1target, test_size=0.5, random_state=random_state)

subscales = ["FBI", "MSFUP", "MSFUaprivate", "MSFUapublic", "COMF", "CSS", "RSES", "RRS"]

scales = [col for col in Data1.columns if not col in subscales]

X_train_subscales = X_train.drop(scales, axis=1)
X_train_scales = X_train.drop(subscales, axis=1)
X_val_subscales = X_val.drop(scales, axis=1)
X_val_scales = X_val.drop(subscales, axis=1)
X_test_subscales = X_test.drop(scales, axis=1)
X_test_scales = X_test.drop(subscales, axis=1)


data = [X_train_scales, X_train_subscales, y_train, X_val_scales, X_val_subscales, y_val, X_test_scales, X_test_subscales, y_test]
names = ["X_train_scales", "X_train_subscales", "y_train", "X_val_scales", "X_val_subscales", "y_val", "X_test_scales", "X_test_subscales", "y_test"]
for i, component in enumerate(data):
	component.to_pickle(f"./scale_data/{names[i]}.pkl")