import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline

X_train = pd.read_pickle('./subscale_data/X_train.pkl')
X_val = pd.read_pickle('./subscale_data/X_val.pkl')
y_train = pd.read_pickle('./subscale_data/y_train.pkl')
y_val = pd.read_pickle('./subscale_data/y_val.pkl')

models = {
	'Random forest regressor': make_pipeline(
		RandomForestRegressor(1000, max_features='sqrt', random_state=42)
	),
	'Decision tree regressor': make_pipeline(
		DecisionTreeRegressor()
	),
	'Ridge': make_pipeline(
		Ridge()
	),
	'Lasso': make_pipeline(
		Lasso()
	),
	'ElasticNet': make_pipeline(
		ElasticNet()
	),
	'Support vector machine': make_pipeline(
		SVR()
	)
}

param_grids = {
	'Random forest regressor': {
		'randomforestregressor__max_depth': [*np.linspace(1, 21, 21).astype(int), None]
	},
	'Decision tree regressor': {
		'decisiontreeregressor__max_depth': [*np.linspace(1, 21, 21).astype(int), None]
	},
	'Ridge': {
		'ridge__alpha': np.logspace(-3, 3, 7)
	},
	'Lasso': {
		'lasso__alpha': np.logspace(-3, 3, 7)
	},
	'ElasticNet': {
		"elasticnet__alpha": np.logspace(-3, 3, 7),
		"elasticnet__l1_ratio": np.linspace(0, 1, 11),
	},	
	'Support vector machine': {
		"svr__kernel": ["linear", "poly", "rbf", "sigmoid"]
	}
}

model_name = "Support vector machine"
model = models[model_name]

grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (MSE):", -grid_search.best_score_)

y_val_pred = grid_search.predict(X_val)

print("MSE:", np.mean((y_val_pred - y_val) ** 2))

# Calculate R^2 for the Support Vector Machine model
r2_svm = r2_score(y_val, y_val_pred)

print("R^2 for Support Vector Machine:", r2_svm)