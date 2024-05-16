import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
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

names = ['summary scores', "individual items"]

for i, scale in enumerate(scales):

	X_train = pd.read_pickle(f'../scale_data/X_train_{scale}.pkl')
	X_val = pd.read_pickle(f'../scale_data/X_val_{scale}.pkl')
	X_test = pd.read_pickle(f'../scale_data/X_test_{scale}.pkl')
	y_train = pd.read_pickle(f'../scale_data/y_train.pkl')
	y_val = pd.read_pickle(f'../scale_data/y_val.pkl')
	y_test = pd.read_pickle(f'../scale_data/y_test.pkl')

	maximum = y_test.max().max()

	X_train = pd.concat([X_train, X_val])
	y_train = pd.concat([y_train, y_val])
	for DASS in DASSes:
		model = optimals[f"{DASS} {scale}"]
		model.fit(X_train, y_train[DASS])
		y_pred = model.predict(X_test)
		mse = mean_squared_error(y_test[DASS], y_pred)
		r2 = r2_score(y_test[DASS], y_pred)
		print(f"Scale: {scale}, DASS: {DASS}, MSE: {mse}, R2: {r2}")
		
		plt.figure(figsize=(8,8))
		plt.scatter(y_test[DASS], y_pred, label=f"R2={round(r2, 2)}")
		plt.plot([0, maximum], [0, maximum], 'k--', label='Identity')
		plt.xlabel('Observed value')
		plt.ylabel('Predicted value')

		linear_model = LinearRegression()
		linear_model.fit(y_test[DASS].to_numpy().reshape(-1, 1), y_pred)
		y_pred_best_fit = linear_model.predict(y_test[DASS].to_numpy().reshape(-1, 1))
		plt.plot(y_test[DASS], y_pred_best_fit, 'k', label='Best fit')

		plt.legend()

		plt.title(f'Prediction error for {DASS.replace("_", " ")} for the {names[i]}')

		# plt.xlim([-5,20])
		plt.ylim([-5,25])

		plt.grid(True, which='both')
		# plt.minorticks_on()  # Enable minor ticks

		# # You can adjust the ticks if necessary, for example:
		# plt.xticks(np.arange(0, maximum+1, 5))
		# plt.yticks(np.arange(0, maximum+1, 5))

		plt.savefig(f'../figures/{DASS}_{names[i]}')
		