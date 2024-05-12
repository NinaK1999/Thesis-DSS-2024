import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

Data1_norm = np.load("./Data1_norm.npy")
Data2_norm = np.load("./Data2_norm.npy")
Data1 = pd.read_csv("./Data1_subscales.csv", delimiter=";")
Data2 = pd.read_csv("./Data2_subscales.csv", delimiter=";")

Data1_norm = pd.DataFrame(Data1_norm, columns=Data1.columns)
Data2_norm = pd.DataFrame(Data2_norm, columns=Data2.columns)

random_state = 42

# Split Data2 into train and Data1 into validation and test sets
X_train, y_train = Data2_norm.drop('Stress', axis=1), Data2_norm['Stress']
X_val, X_test, y_val, y_test = train_test_split(Data1_norm.drop('Stress', axis=1), Data1_norm['Stress'], test_size=0.5, random_state=random_state)

data = [X_train, y_train, X_val, y_val, X_test, y_test]
names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
for i, component in enumerate(data):
	component.to_pickle(f"./subscale_data/{names[i]}.pkl")