import numpy as np


class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)


    def forward(self, X):
        # forward propagation through the network
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)

        return o


    def backward(self, X, y, o):
        # backward propagation through the network
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)


    def train_AI(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


    def predict(self, xPredicted):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))


    def sigmoid(self, s):
        # activation function
        return 1 / ( 1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
