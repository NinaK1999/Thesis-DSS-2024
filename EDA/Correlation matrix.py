import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Data1_norm = np.load("./All_data/Data1_norm.npy")
Data2_norm = np.load("./All_data/Data2_norm.npy")
Data1 = pd.read_csv("./All_data/Data1_subscales.csv", delimiter=";")
Data2 = pd.read_csv("./All_data/Data2_subscales.csv", delimiter=";")

Data1_norm = pd.DataFrame(Data1_norm, columns=Data1.columns)
Data2_norm = pd.DataFrame(Data2_norm, columns=Data2.columns)

# Compute correlation matrix for Data1_norm using Spearman's correlation
correlation_matrix1 = Data1_norm.corr(method='spearman')

# Set up the matplotlib figure for Data1_norm
plt.figure(figsize=(10, 8))
heatmap1 = sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Spearman Correlation Matrix for Data1_norm')

# Rotate the x-axis tick labels at 45 degrees
plt.xticks(rotation=45)

plt.show()

# Compute correlation matrix for Data2_norm using Spearman's correlation
correlation_matrix2 = Data2_norm.corr(method='spearman')

# Set up the matplotlib figure for Data2_norm
plt.figure(figsize=(10, 8))
heatmap2 = sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Spearman Correlation Matrix for Data2_norm')

# Rotate the x-axis tick labels at 45 degrees
plt.xticks(rotation=45)

plt.show()
