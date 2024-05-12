import pandas as pd
import numpy as np

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

import warnings

X_train = pd.read_pickle('./scale_data/X_train.pkl')
X_val = pd.read_pickle('./scale_data/X_val.pkl')
X_test = pd.read_pickle('./scale_data/X_test.pkl')
y_train = pd.read_pickle('./scale_data/y_train.pkl')
y_val = pd.read_pickle('./scale_data/y_val.pkl')
y_test = pd.read_pickle('./scale_data/y_test.pkl')

models = {
	'Random forest regressor': make_pipeline(
		RandomForestRegressor(1000, max_features='sqrt', random_state=42)
	),
	'Random forest regressor fitted': make_pipeline(
		RandomForestRegressor(1000, max_features='sqrt', random_state=42, max_depth=12)
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
		ElasticNet(max_iter=10000)
	),
	'Support vector machine': make_pipeline(
		SVR()
	),
	'PCA + Ridge': make_pipeline(
		PCA(),
		Ridge(0.01),
	),
	"PCA + random forest regressor": make_pipeline(
		PCA(), 
		RandomForestRegressor(1000, max_depth=12, max_features='sqrt', random_state=42)
	),
	"SelectFromModel + random forest regressor": make_pipeline(
		SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42)),
		RandomForestRegressor(1000, max_depth=12, max_features='sqrt', random_state=42)
	),
	"SelectFromModel + random forest regressor fitted": make_pipeline(
		SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42)),
		RandomForestRegressor(1000, max_depth=12, max_features='sqrt', random_state=42)
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
	},
	'PCA + Ridge': {
		"pca__n_components": np.linspace(1, X_train.shape[1], X_train.shape[1]).astype(int),
	},
	'PCA + random forest regressor': {
		"pca__n_components": np.linspace(1, X_train.shape[1], X_train.shape[1]).astype(int),
		'randomforestregressor__max_depth': [*np.linspace(1, 21, 21).astype(int), None]
	},
	"SelectFromModel + random forest regressor": {
		"selectfrommodel__max_features": np.linspace(1, X_train.shape[1], X_train.shape[1]).astype(int),
		'randomforestregressor__max_depth': [*np.linspace(1, 21, 21).astype(int), None]
	}

}


data = {
    "Model": ["Random forest regressor", "Decision tree regressor", "PCA + random forest regressor"],
    "DASS_anxiety Optimal parameters": ["max. depth=12", "", "n_components"],
    "DASS_anxiety Training score MSE": ["MSE_value", "MSE_value", "MSE_value"], 
    "DASS_anxiety Training score R2": ["R2_value", "R2_value", "R2_value"],
    "DASS_anxiety Validation score MSE": ["MSE_value", "MSE_value", "MSE_value"], 
    "DASS_anxiety Validation score R2": ["R2_value", "R2_value", "R2_value"],
    "DASS_stress Optimal parameters": ["max. depth=12", "", "n_components"],
    "DASS_stress Training score MSE": ["MSE_value", "MSE_value", "MSE_value"],
    "DASS_stress Training score R2": ["R2_value", "R2_value", "R2_value"],
    "DASS_stress Validation score MSE": ["MSE_value", "MSE_value", "MSE_value"],
    "DASS_stress Validation score R2": ["R2_value", "R2_value", "R2_value"],
    "DASS_depression Optimal parameters":  ["max. depth=12", "", "n_components"],
    "DASS_depression Training score MSE": ["MSE_value", "MSE_value", "MSE_value"],
    "DASS_depression Training score R2": ["R2_value", "R2_value", "R2_value"],
    "DASS_depression Validation score MSE": ["MSE_value", "MSE_value", "MSE_value"], 
    "DASS_depression Validation score R2": ["R2_value", "R2_value", "R2_value"],
}

for key in data:
	data[key] = []

warnings.filterwarnings('ignore', category=ConvergenceWarning)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
# print(models.keys()[])
for model_name in list(models.keys())[2:4]:
	model = models[model_name]

	data["Model"].append(model_name)

	for column in y_train:
		print(column)
		y_train_column, y_val_column = y_train[column], y_val[column]
		grid_search = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='neg_mean_squared_error', verbose=0)
		grid_search.fit(X_train, y_train_column)

		best_params = grid_search.best_params_
		data[f"{column} Optimal parameters"].append(", ".join([f"{parameter} = {best_params[parameter]}" for parameter in best_params]))
		# print("Best cross-validation score (MSE):", -grid_search.best_score_)

		# Predictions on the training and validation sets
		y_train_pred = grid_search.predict(X_train)
		y_val_pred = grid_search.predict(X_val)

		# Calculate MSE for the validation set
		mse_train = mean_squared_error(y_train_column, y_train_pred)
		mse_val = mean_squared_error(y_val_column, y_val_pred)

		data[f"{column} Training score MSE"].append(mse_train)
		data[f"{column} Validation score MSE"].append(mse_val)

		# Calculate R2 for the training and validation sets
		r2_train = r2_score(y_train_column, y_train_pred)
		r2_val = r2_score(y_val_column, y_val_pred)

		data[f"{column} Training score R2"].append(r2_train)
		data[f"{column} Validation score R2"].append(r2_val)


df = pd.DataFrame(data)

df.columns = pd.MultiIndex.from_tuples([
    ('Model',"",""), 
    ('DASS_anxiety', 'Optimal parameters', ""), 
    ('DASS_anxiety', 'Training scores', 'MSE'), ('DASS_anxiety', 'Training scores', 'R2'),
    ('DASS_anxiety', 'Validation scores', 'MSE'), ('DASS_anxiety', 'Validation scores', 'R2'),
    ('DASS_stress', 'Optimal parameters', ""), 
    ('DASS_stress', 'Training scores', 'MSE'), ('DASS_stress', 'Training scores', 'R2'),
    ('DASS_stress', 'Validation scores', 'MSE'), ('DASS_stress', 'Validation scores', 'R2'),
    ('DASS_depression', 'Optimal parameters', ""), 
    ('DASS_depression', 'Training scores', 'MSE'), ('DASS_depression', 'Training scores', 'R2'),
    ('DASS_depression', 'Validation scores', 'MSE'), ('DASS_depression', 'Validation scores', 'R2'),
])


# Write DataFrame to Excel file
df.to_excel('model_comparison.xlsx', engine='openpyxl', index=True)

# Write DataFrame to CSV file
# df.to_csv('/mnt/data/model_comparison.csv', index=False)

print("Files saved: model_comparison.xlsx")

# model = models['SelectFromModel + random forest regressor fitted']
# model.fit(X_train, y_train)

# # Get the fitted SelectFromModel from the pipeline
# select_from_model = model.named_steps['selectfrommodel']

# # Get the support mask for the selected features
# selected_features_mask = select_from_model.get_support()

# # Print which features are selected
# print("Selected features:", X_train.columns[selected_features_mask])

# y_val_pred = model.predict(X_val)

# print("MSE:", np.mean((y_val_pred-y_val)**2))

# model = models['Random forest regressor fitted']
# model.fit(X_train, y_train)

# y_val_pred = model.predict(X_val)

# print("MSE:", np.mean((y_val_pred-y_val)**2))