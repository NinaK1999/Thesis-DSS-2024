import pandas as pd
import numpy as np
from scipy import stats
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

import pandas as pd
import numpy as np

Data1 = pd.read_excel('./Original data/dataset_study1_OSF.xlsx')
Data2 = pd.read_excel('./Original data/dataset_study1_OSF.xlsx')

filtered_columns = Data1.filter(regex='\d$').filter(regex='([^FBI8])').filter(regex='([^FBI7])').filter(regex='FBI')

n_respondents = filtered_columns.shape[0]
n_scale_items = filtered_columns.shape[1]
n_categories = max(filtered_columns.max()) - min(filtered_columns.min()) + 1

scaleData = np.array(filtered_columns.values.tolist())
frequencies = np.array([[0 for _ in range(n_categories + 1)] for _ in range(n_scale_items + 1)])
totalForNj = np.array([0 for _ in range(n_categories + 1)])
totalForGi = np.array([0 for _ in range(n_scale_items + 1)])
accumulativeFrequencies = np.array([[0 for _ in range(n_categories + 1)] for _ in range(n_scale_items + 1)])

accumulativeProportions = [[0 for _ in range(n_categories + 1)] for _ in range(n_scale_items + 1)]
interval = [0 for _ in range(n_categories)]
X = [0 for _ in range(n_categories + 1)]
scores = [0 for _ in range(n_categories + 1)]

for r in range(1, n_respondents + 1):
	for s in range(1, n_scale_items + 1):
		response = scaleData[r - 1][s - 1]
		frequencies[s][response] += 1

for c in range(1, n_categories + 1):
	for s in range(1, n_scale_items + 1):
		totalForNj[c] += frequencies[s][c]

for s in range(1, n_scale_items + 1):
	for c in range(1, n_categories + 1):
		totalForGi[s] += frequencies[s][c]

for s in range(1, n_scale_items + 1):
	for c in range(1, n_categories + 1):
		if c == 1:
			accumulativeFrequencies[s][c] = frequencies[s][c]
		else:
			accumulativeFrequencies[s][c] = accumulativeFrequencies[s][c - 1] + frequencies[s][c]
print(accumulativeFrequencies)

for c in range(1, n_categories + 1):
	for s in range(1, n_scale_items + 1):
		value = accumulativeFrequencies[s, c]/totalForGi[s]
		accumulativeProportions[s][c] = value

accumulativeProportions = np.array(accumulativeProportions)

k = n_categories
c = n_categories - 1
j = c

m = n_scale_items

frequencies = np.array(frequencies)
accumulativeProportions = np.array(accumulativeProportions)

def F1(j):
	return np.log(totalForNj[k-1] / (sum((frequencies[1:, k - 1] + frequencies[1:,k])*accumulativeProportions[1:, k - 1]) - totalForNj[k-1]) + 1)

def F2(j):
	return np.log(totalForNj[j-1] / (sum((frequencies[1:, j] + frequencies[1:,j + 1])*accumulativeProportions[1:, j]) + totalForNj[j + 1]/(np.exp(interval[j + 1]) - 1) - totalForNj[j]) + 1)

for c in range(n_categories - 1, 1, -1):
	if c == n_categories - 1:
		interval[c] = F1(c)
	else:
		interval[c] = F2(c)

X[0] = -np.inf
X[1] = 0
X[n_categories] = np.inf

for c in range(2, n_categories):
	X[c] = X[c-1] + interval[c]

for c in range(2, n_categories):
	scores[c] = (X[c] + X[c - 1])/2

if np.mean(accumulativeProportions[1:, 1]) < 0.1:
	scores[1] = X[1] - 1
else:
	scores[1] = X[1] - 1.1

if np.mean(accumulativeProportions[1:, -1] - accumulativeProportions[1:, -2]) < 0.1:
	scores[k] = X[k - 1] + 1
else:
	scores[k] = X[k - 1] + 1.1

result = [[None for _ in range(n_scale_items)] for _ in range(n_respondents)]

def map_value_to_score(value):
	for i in range(len(X) - 1):
		if X[i] <= value and value < X[i + 1]:
			return scores[i + 1]


for i in range(n_scale_items):
	for j in range(n_respondents):
		result[j][i] = map_value_to_score(scaleData[j][i])

# vectorized_func = np.vectorize(map_value_to_score)

res = stats.shapiro(result)
print("Shapiro transformed:", res.statistic)
res = stats.shapiro(scaleData)
print("Shapiro original:", res.statistic)

# # Use nonparanormal normalization from 'huge' library from Rstudio
pandas2ri.activate()
robjects.r('library("huge")')

# Data1 = Data1.filter(regex='\d$').filter(regex='([^FBI8])').filter(regex='([^FBI7])').filter(regex='FBI')
Data1_norm = pd.DataFrame(robjects.r['huge.npn'](Data1), columns=Data1.columns)
Data2_norm = pd.DataFrame(robjects.r['huge.npn'](Data2), columns=Data2.columns)


values = Data1_norm.values.tolist()
res = stats.shapiro(values)
print("Shapiro npn:", res.statistic)
