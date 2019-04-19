import numpy as np


class NNMultiClassClassifier:

    def __init__(self, hidden_nodes, output_labels, epoch, lr):
        self.hidden_nodes = hidden_nodes
        self.output_labels = output_labels
        self.epoch = epoch
        self.lr = lr

        self.wo = np.random.rand(self.hidden_nodes, self.output_labels)
        self.bo = np.random.randn(self.output_labels)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def fit(self, X, one_hot_y):
        attributes = X.shape[1]

        self.wh = np.random.rand(attributes, self.hidden_nodes)
        self.bh = np.random.randn(self.hidden_nodes)

        for i in range(self.epoch):
            # feedforward

            # Phase 1
            zh = np.dot(X, self.wh) + self.bh
            ah = self.sigmoid(zh)

            # Phase 2
            zo = np.dot(ah, self.wo) + self.bo
            ao = self.softmax(zo)

            # Back Propagation

            # Phase 1

            dcost_dzo = ao - one_hot_y
            dzo_dwo = ah

            dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

            dcost_bo = dcost_dzo

            # Phases 2

            dzo_dah = self.wo
            dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
            dah_dzh = self.sigmoid_der(zh)
            dzh_dwh = X
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

            dcost_bh = dcost_dah * dah_dzh

            # Update Weights ================

            self.wh -= self.lr * dcost_wh
            self.bh -= self.lr * dcost_bh.sum(axis=0)

            self.wo -= self.lr * dcost_wo
            self.bo -= self.lr * dcost_bo.sum(axis=0)

            if i % 200 == 0:
                loss = np.sum(-one_hot_y * np.log(ao))
                print('Loss function value: ', loss)

    def score(self, X, one_hot_y):
        zh = np.dot(X, self.wh) + self.bh
        ah = self.sigmoid(zh)

        zo = np.dot(ah, self.wo) + self.bo
        ao = self.softmax(zo)

        results = np.zeros(ao.shape)
        for i in range(ao.shape[0]):
            idx = np.argmax(ao[i])
            results[i][idx] = 1

        correct_num = 0
        for result, label in zip(results, one_hot_y):
            correct = result == label
            if correct.all():
                correct_num += 1

        return float(correct_num) / int(results.shape[0])
