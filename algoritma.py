import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('clean_table.csv')

X = np.array(data.drop(columns = ['game_conclusion']))
y = data['game_conclusion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
feature_set = np.array(X_train)
output = np.array(y_train)
labels = [0] * feature_set.shape[0]

for i in range(len(labels)):
    label = output[i]
    if label == "HOME":
        labels[i] = 0
    elif label == "DRAW":
        labels[i] = 1
    else:
        labels[i] = 2
        
one_hot_labels = np.zeros((len(labels), 3))

for i in range(len(labels)):  
    one_hot_labels[i, labels[i]] = 1
    
def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = feature_set.shape[0]  
attributes = feature_set.shape[1]  
hidden_nodes = 7
output_labels = 3

wh = np.random.rand(attributes,hidden_nodes)  
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)  
bo = np.random.randn(output_labels)  
lr = 10e-4

for epoch in range(5000):  
############# feedforward

    # Phase 1
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    # Phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

########## Back Propagation

########## Phase 1

    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)
    
    if epoch % 200 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        print('Loss function value: ', loss)
        

results = np.zeros(ao.shape)
for i in range(ao.shape[0]):
    idx = np.argmax(ao[i])
    results[i][idx] = 1
    
correct_num = 0
for result,label in zip(results,one_hot_labels):
    correct = result == label
    if correct.all():
        correct_num += 1
        
print("accuracy: ", float(correct_num)/int(results.shape[0]))