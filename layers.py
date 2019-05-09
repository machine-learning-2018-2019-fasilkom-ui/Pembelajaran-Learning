import numpy as np


class Relu:
    def forward(self, _input):
        return np.maximum(0, _input)

    def backward(self, _input, grad_output):
        relu_grad = np.array(_input) > 0
        return np.array([grad_output * relu_grad])


class Sigmoid:
    def forward(self, _input):
        return np.array([1 / (1 + np.exp(-x)) for x in _input])

    def backward(self, _input, grad_output):
        grad_input = [np.e ** -x / (np.e ** -x + 1) ** 2 for x in _input]
        return np.array([grad_input])


class Softmax:
    def forward(self, _input):
        return np.array([np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))) for x in _input])

    def backward(self, _input, grad_output):
        return np.array([grad_output])


class Dense:
    def __init__(self, input_shape, output_shape):

        # xavier weight initializer
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape))
        self.bias = np.random.randn(output_shape)

    def forward(self, _input):
        return np.array([np.add(wa, self.bias) for wa in np.matmul(_input, self.weights)])

    def backward(self, _input, grad_output):
        _input = np.array(_input)
        grad_output = np.array(grad_output)
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(_input.T, grad_output)
        grad_bias = grad_output.mean(axis=0) * _input.shape[0]

        return [grad_input, grad_weights, grad_bias]
