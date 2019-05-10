import numpy as np

def encode(y):
    labels = [0] * y.shape[0]
    for i in range(len(labels)):
        label = y[i]
        if label == "HOME":
            labels[i] = 0
        elif label == "DRAW":
            labels[i] = 1
        else:
            labels[i] = 2
            
    one_hot_label = np.zeros((len(labels),3))
    for i in range(len(labels)):
        one_hot_label[i, labels[i]] = 1
    
    return one_hot_label