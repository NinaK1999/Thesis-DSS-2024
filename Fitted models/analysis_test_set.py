import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

optimals = {
	"DASS_anxiety subscales": make_pipeline(
		SVR(kernel='linear')
	),
	"DASS_stress subscales": make_pipeline(
		SVR(kernel='linear')
	),
	"DASS_depression subscales": make_pipeline(
		Lasso(alpha=0.1)
	),
	"DASS_anxiety scales": make_pipeline(
		PCA(n_components=60),
		Ridge(alpha=100),
	),
	"DASS_stress scales": make_pipeline(
		SVR(kernel='sigmoid')
	),
	"DASS_depression scales": make_pipeline(
		Lasso(alpha=0.1)
	),
}

DASSes = ["DASS_anxiety", "DASS_stress", "DASS_depression"]

scales = ["subscales", "scales"]

for scale in scales:

	X_train = pd.read_pickle(f'../scale_data/X_train_{scale}.pkl')
	X_val = pd.read_pickle(f'../scale_data/X_val_{scale}.pkl')
	X_test = pd.read_pickle(f'../scale_data/X_test_{scale}.pkl')
	y_train = pd.read_pickle(f'../scale_data/y_train.pkl')
	y_val = pd.read_pickle(f'../scale_data/y_val.pkl')
	y_test = pd.read_pickle(f'../scale_data/y_test.pkl')

	X_train = pd.concat([X_train, X_val])
	y_train = pd.concat([y_train, y_val])
	for DASS in DASSes:
		model = optimals[f"{DASS} {scale}"]
		model.fit(X_train, y_train[DASS])
		y_pred = model.predict(X_test)
		mse = mean_squared_error(y_test[DASS], y_pred)
		r2 = r2_score(y_test[DASS], y_pred)
		print(f"Scale: {scale}, DASS: {DASS}, MSE: {mse}, R2: {r2}")