from ANNClassifier import ANNClassifier
from one_hot_encoder import encode
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np

data = pd.read_csv('clean_table.csv')

# Extract feature and label
X = np.array(data.drop(columns=['game_conclusion']))
y = data['game_conclusion']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


x_train = np.array(x_train)
one_hot_train = encode(np.array(y_train))

x_test = np.array(x_test)
one_hot_test = encode(np.array(y_test))

activation='relu'
hidden_layer_sizes=(108,)

network = ANNClassifier(epoch=20, hidden_layer_sizes=hidden_layer_sizes, activation=activation)
network.fit(x_train, one_hot_train, batch_size=10)

clf = MLPClassifier(activation=activation,hidden_layer_sizes=hidden_layer_sizes)
clf.fit(x_train, one_hot_train)

result = network.predict_proba(x_test)
print("Our model accuracy:",network.score(x_test, one_hot_test))
print("Sklearn model accuracy:",clf.score(x_test, one_hot_test))