import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLassoCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline

X_train = pd.read_pickle('./scale_data/total_train.pkl')

edge_model = GraphicalLassoCV(cv=5)

edge_model.fit(X_train)
print(X_train.head())

p = edge_model.covariance_

sns.heatmap(p)

sns.heatmap(p, cmap="coolwarm", square=True)

plt.xticks(ticks=range(len(X_train.columns)), labels=X_train.columns, rotation=45)
plt.yticks(ticks=range(len(X_train.columns)), labels=X_train.columns, rotation=0)

plt.title('Precision Matrix Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()