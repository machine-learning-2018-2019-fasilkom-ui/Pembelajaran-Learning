from NNMultiLayer import NNMultiLayer
from one_hot_encoder import encode
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('clean_table.csv')

# Extract feature and label
X = np.array(data.drop(columns=['game_conclusion']))
y = data['game_conclusion']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


x_train = np.array(x_train)
one_hot_train = encode(np.array(y_train))

x_test = np.array(x_test)
one_hot_test = encode(np.array(y_test))

activation = 'relu'
epochs = [25, 50, 100, 150, 200, 250, 300]
hidden_layer_sizes = (128,)

for epoch in epochs:
    network = NNMultiLayer(epoch=epoch, hidden_layer_sizes=hidden_layer_sizes, activation=activation)
    losses = network.fit(x_train, one_hot_train, rand_seed=37)
    print("epoch:",epoch)
    print("Our model accuracy:", network.score(x_test, one_hot_test))
    total_iter = list(range(len(losses)))
    plt.plot(total_iter, losses, 'b')
    plt.grid()
    plt.title("Epoch: {:d}, Activation: {:s}, Hidden Layers: {:s}".format(epoch, activation, str(hidden_layer_sizes)))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    clf = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes,max_iter=epoch)
    clf.fit(x_train, one_hot_train)
    
    print("Sklearn model accuracy:",clf.score(x_test, one_hot_test))
    print("\n")