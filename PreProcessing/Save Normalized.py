import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np

# Use subscales file Data1
Data1 = pd.read_csv("./Data1_subscales.csv", delimiter=";")

# Use nonparanormal normalization from 'huge' library from Rstudio
pandas2ri.activate()
robjects.r('library("huge")')

Data1_norm = robjects.r['huge.npn'](Data1)
# Data1_norm = pd.DataFrame(Data1_norm, columns=Data1.columns)

np.save("/Users/ninakleinlugtebeld/Library/Mobile Documents/com~apple~CloudDocs/Data Science and Society/Block 3/Thesis/Data/Data1_norm.npy", Data1_norm)

# Use subscales file Data2
Data2 = pd.read_csv("./Data2_subscales.csv", delimiter=";")

# Use nonparanormal normalization from 'huge' library from Rstudio
pandas2ri.activate()
robjects.r('library("huge")')

Data2_norm = robjects.r['huge.npn'](Data2)
# Data2_norm = pd.DataFrame(Data2_norm, columns=Data2.columns)

np.save("/Users/ninakleinlugtebeld/Library/Mobile Documents/com~apple~CloudDocs/Data Science and Society/Block 3/Thesis/Data/Data2_norm.npy", Data2_norm)