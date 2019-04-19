from sklearn.model_selection import train_test_split
from NNMultiClassClassifier import NNMultiClassClassifier

import pandas as pd
import numpy as np

network = NNMultiClassClassifier(7, 3, 500, 10e-4)
data = pd.read_csv('clean_table.csv')

X = np.array(data.drop(columns=['game_conclusion']))
y = data['game_conclusion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model fitting
X_train = np.array(X_train)
y_train = np.array(y_train)

labels = [0] * X_train.shape[0]
for i in range(len(labels)):
    label = y_train[i]
    if label == "HOME":
        labels[i] = 0
    elif label == "DRAW":
        labels[i] = 1
    else:
        labels[i] = 2
        
one_hot_labels = np.zeros((len(labels), 3))
for i in range(len(labels)):
    one_hot_labels[i, labels[i]] = 1

network.fit(X_train, one_hot_labels)

# Model testing
X_test = np.array(X_test)
y_test = np.array(y_test)
labels = [0] * X_test.shape[0]

for i in range(len(labels)):
    label = y_test[i]
    if label == "HOME":
        labels[i] = 0
    elif label == "DRAW":
        labels[i] = 1
    else:
        labels[i] = 2

one_hot_labels = np.zeros((len(labels), 3))

for i in range(len(labels)):
    one_hot_labels[i, labels[i]] = 1

print("Accuracy: ", network.score(X_test, one_hot_labels))