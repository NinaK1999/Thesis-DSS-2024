# Importing required libraries
import pandas as pd
from scipy.stats import spearmanr

# Load the data
X_train = pd.read_pickle('./scale_data/X_train.pkl')
X_val = pd.read_pickle('./scale_data/X_val.pkl')
y_train = pd.read_pickle('./scale_data/y_train.pkl')
y_val = pd.read_pickle('./scale_data/y_val.pkl')

# Compute Spearman's correlation for X_train
corr_matrix_X_train = X_train.corr(method='spearman')

# Compute Spearman's correlation for X_val
corr_matrix_X_val = X_val.corr(method='spearman')

# Display the correlation matrices as tables
print("Spearman's correlation matrix for X_train:")
print(corr_matrix_X_train)

print("\nSpearman's correlation matrix for X_val:")
print(corr_matrix_X_val)
