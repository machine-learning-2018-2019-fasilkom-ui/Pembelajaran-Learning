from ANNClassifier import ANNClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

data = pd.read_csv('clean_table.csv')

X = np.array(data.drop(columns=['game_conclusion']))
y = data['game_conclusion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model fitting
X_train = np.array(X_train)
y_train = np.array(y_train)

train_labels = [0] * X_train.shape[0]
for i in range(len(train_labels)):
    label = y_train[i]
    if label == "HOME":
        train_labels[i] = 0
    elif label == "DRAW":
        train_labels[i] = 1
    else:
        train_labels[i] = 2
        
one_hot_train = np.zeros((len(train_labels), 3))
for i in range(len(train_labels)):
    one_hot_train[i, train_labels[i]] = 1

network = ANNClassifier(500, hidden_layer_sizes=(20, 20))
network.fit(X_train, one_hot_train)

# Model testing
X_test = np.array(X_test)
y_test = np.array(y_test)
test_labels = [0] * X_test.shape[0]

for i in range(len(test_labels)):
    label = y_test[i]
    if label == "HOME":
        test_labels[i] = 0
    elif label == "DRAW":
        test_labels[i] = 1
    else:
        test_labels[i] = 2

one_hot_test = np.zeros((len(test_labels), 3))

for i in range(len(test_labels)):
    one_hot_test[i, test_labels[i]] = 1

print(network.predict_proba(X_test))