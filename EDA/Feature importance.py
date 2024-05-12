from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn.tree import DecisionTreeRegressor

Data1_norm = np.load("./Data1_norm.npy")
Data2_norm = np.load("./Data2_norm.npy")
Data1 = pd.read_csv("./Data1_subscales.csv", delimiter=";")
Data2 = pd.read_csv("./Data2_subscales.csv", delimiter=";")

Data1_norm = pd.DataFrame(Data1_norm, columns=Data1.columns)
Data2_norm = pd.DataFrame(Data2_norm, columns=Data2.columns)

X = Data1_norm.drop('Depression', axis=1) # Features
y = Data1_norm['Depression']  # Target

# Implement
def rf_feature_importance(X, y):
    """ Compute the feature importances for the given data as described above.
    Args:
        X: the input features
        y: the target variable
    Returns:
        A tuple containing the feature names and their importances, sorted by importance.
    """

    # Creating a pipeline for preprocessing and model fitting
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=0, max_features="sqrt", n_jobs=-1))
    ])
    
    # Fitting the model
    pipeline.fit(X, y)
    
    # Getting feature importances
    importances = pipeline.named_steps['rf'].feature_importances_

    # Sorting the data in increasing order
    indices = np.flip(importances.argsort(), 0)

    return X.columns[indices], importances[indices]

def plot_feature_importance(features,importances):
    """ Plot the features and their importances as a bar chart in order of importance.
    Args:
        features: the feature names, ordered by importance
        importances: the feature importances
    """
    n_features = len(features)
    plt.figure(figsize=(7,5.4))
    plt.barh(range(n_features), importances, align='center')
    plt.yticks(np.arange(n_features), features, fontsize=7)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()

importances = rf_feature_importance(X, y)
plot_feature_importance(*importances)
