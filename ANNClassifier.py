from layers import Softmax, Sigmoid, Dense, Relu
import numpy as np


class ANNClassifier:
    def __init__(self, epoch, beta=0.9, learning_rate=1e-4, hidden_layer_sizes=(50, 50)):
        self.epoch = epoch
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def _create_model(self, input_nodes, output_nodes):
        self.model = []
        self.model.append(Dense(input_nodes, self.hidden_layer_sizes[0], 'xavier'))
        self.model.append(Relu())
        np.random.seed(2017)
        for i in range(len(self.hidden_layer_sizes)-1):
            self.model.append(Dense(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1], 'xavier'))
            self.model.append(Sigmoid())
        np.random.seed(3011)
        self.model.append(Dense(self.hidden_layer_sizes[-1], output_nodes, 'xavier'))
        self.model.append(Softmax())

    def _loss_function(self, layer_input_output_cache, y_true):
        logits = layer_input_output_cache[-1]
        y_pred = Softmax().forward(logits)
        softmax_cross_entropy_loss = -1.0 / len(y_true) * np.sum(
            [y_true[i] * np.log(y_pred[i]) for i in range(len(y_true))])
        softmax_cross_entropy_grad = y_pred - y_true
        return softmax_cross_entropy_loss, softmax_cross_entropy_grad

    def _update_layer(self, layer, weight, bias):
        if id(layer) not in self.velocity:
            self.velocity[id(layer)] = 0

        self.velocity[id(layer)] = self.beta * self.velocity[id(layer)] + (1 - self.beta) * weight
        layer.weights = layer.weights - self.learning_rate * self.velocity[id(layer)]
        layer.bias = layer.bias - self.learning_rate * bias

    # forward propagation    
    def _forward(self, _input):
        # cache the input and output to use later
        layer_input_output_cache = [_input]

        # forward propagate for each layer
        for layer in self.model:
            _input = layer.forward(_input)
            layer_input_output_cache.append(_input)

        return layer_input_output_cache

    def _backward(self, layer_input_output_cache, output_grad):
        # reverse loop
        for j in range(len(self.model))[::-1]:

            layer = self.model[j]

            # list of gradient
            backprop = layer.backward(layer_input_output_cache[j], output_grad)

            # update output grad and
            if len(backprop) > 1:
                # if contain weight and bias update this layer bias and weight
                [output_grad, weight, bias] = backprop
                self._update_layer(layer, weight, bias)
            else:
                output_grad = backprop[0]

    def fit(self, x_train, y_train, x_validate=None, y_validate=None, batch_size=None):
        self._create_model(x_train.shape[1], y_train.shape[1])
        if batch_size is None:
            batch_size = x_train.shape[0]

        total_m_train = x_train.shape[0]

        use_validation = x_validate is not None
        if use_validation:
            total_m_val = x_validate.shape[0]

        # we are using the definition of epoch as 1 iteration to all training example
        history_train = []
        history_val = []

        for epoch in range(self.epoch):

            history_per_batch = []
            for i in (range(0, total_m_train, batch_size)):
                # separate each set to minibatches
                adjusted_batch_size = min(i + batch_size, total_m_train)
                X_batch_train = x_train[i:adjusted_batch_size]
                y_batch_train = y_train[i:adjusted_batch_size]

                # forward propagate
                layer_input_output_cache = self._forward(X_batch_train)

                # compute its loss and grad
                loss, output_grad = self._loss_function(layer_input_output_cache, y_batch_train)
                # report current progress
                history_per_batch.append(loss)

                # backward propagation
                self._backward(layer_input_output_cache, output_grad)
            history_train.append(history_per_batch)

            train_loss_mean = np.mean(history_per_batch)
            print("Train Loss Mean:", train_loss_mean)
            val_loss_mean = None

            if use_validation:
                history_per_batch = []
                for i in (range(0, total_m_val, batch_size)):
                    # separate each set to minibatches
                    adjusted_batch_size = min(i + batch_size, total_m_val)
                    X_batch_val = x_validate[i:adjusted_batch_size]
                    y_batch_val = y_validate[i:adjusted_batch_size]

                    # forward propagate
                    layer_input_output_cache = self._forward(X_batch_val)
                    loss, output_grad = self._loss_function(layer_input_output_cache, y_batch_val)

                    history_per_batch.append(loss)
                history_val.append(history_per_batch)

                val_loss_mean = np.mean(history_per_batch)
            if val_loss_mean is not None:
                print("Validation Loss Mean:", val_loss_mean)

        return history_train, history_val

    def predict_proba(self, X):
        _input = X
        for layer in self.model:
            _input = layer.forward(_input)

        return np.argmax(_input, axis=1).T

    def score(self, x, y):
        correct_num = 0
        for result, label in zip(self.predict_proba(x), np.argmax(y, axis=1)):
            if result == label:
                correct_num += 1

        return float(correct_num) / int(x.shape[0])
