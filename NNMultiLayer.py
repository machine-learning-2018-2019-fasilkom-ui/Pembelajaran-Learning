from layers import Softmax, Sigmoid, Dense, Relu
import numpy as np


class NNMultiLayer:
    def __init__(self, hidden_layer_sizes=(50, 50), epoch=100, activation='relu', lr=1e-4, beta=0.9):
        if activation not in ['relu', 'sigmoid']:
            raise ValueError('activation function must be either relu or sigmoid')
        if activation == 'relu':
            self.activation = Relu
        elif activation == 'sigmoid':
            self.activation = Sigmoid
        self.epoch = epoch
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.beta = beta
        self.velocity = {}

    def _create_model(self, input_nodes, output_nodes, random_seed):
        self.model = []
        np.random.seed(random_seed)
        self.model.append(Dense(input_nodes, self.hidden_layer_sizes[0]))
        self.model.append(self.activation())
        for i in range(len(self.hidden_layer_sizes)-1):
            np.random.seed(random_seed + (i+1) * random_seed)
            self.model.append(Dense(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1]))
            self.model.append(self.activation())
        self.model.append(Dense(self.hidden_layer_sizes[-1], output_nodes))

    def _loss_function(self, layer_input_output_cache, y_true):
        logits = layer_input_output_cache[-1]
        y_pred = Softmax().forward(logits)

        softmax_cross_entropy_loss = -1.0 / len(y_true) * np.sum([y_true[i] * np.log(y_pred[i]) for i in range(len(y_true))])
        softmax_cross_entropy_grad = y_pred - y_true
        return softmax_cross_entropy_loss, softmax_cross_entropy_grad

    def _update_layer(self, layer, weight, bias):
        if id(layer) not in self.velocity:
            self.velocity[id(layer)] = 0

        self.velocity[id(layer)] = self.beta * self.velocity[id(layer)] + (1 - self.beta) * weight
        layer.weights = layer.weights - self.lr * self.velocity[id(layer)]
        layer.bias = layer.bias - self.lr * bias

    # forward propagation
    def _forward(self, _input):
        # cache the input and output to use later
        all_layer_forward_props = [_input]

        # forward propagate for each layer
        for layer in self.model:
            _input = layer.forward(_input)
            all_layer_forward_props.append(_input)

        return all_layer_forward_props

    def _backward(self, layer_input_output_cache, output_grad):
        # iterate from the rightmost layer
        for j in range(len(self.model))[::-1]:

            layer = self.model[j]

            # list of gradient
            backprop = layer.backward(layer_input_output_cache[j], output_grad)

            # update output grad or
            if len(backprop) > 1:
                # if dense layer then update this layer bias and weight
                [output_grad, weight, bias] = backprop
                self._update_layer(layer, weight, bias)
            else:
                output_grad = backprop[0]

    def fit(self, x_train, y_train, rand_seed=2017):
        self._create_model(x_train.shape[1], y_train.shape[1], rand_seed)

        loss_history = []
        for epoch in range(self.epoch):
            # forward propagation
            layer_input_output_cache = self._forward(x_train)

            # compute loss and grad
            loss, output_grad = self._loss_function(layer_input_output_cache, y_train)

            # backward propagation
            self._backward(layer_input_output_cache, output_grad)

            loss_history.append(loss)

        return loss_history

    def predict_proba(self, x):
        _input = x
        for layer in self.model:
            _input = layer.forward(_input)

        return np.argmax(_input, axis=1).T

    def score(self, x, y):
        correct_num = 0
        for result, label in zip(self.predict_proba(x), np.argmax(y, axis=1)):
            if result == label:
                correct_num += 1

        return float(correct_num) / int(x.shape[0])
