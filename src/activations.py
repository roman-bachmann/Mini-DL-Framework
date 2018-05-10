from torch import FloatTensor
from module import Module
import math


def relu(input, inplace=False):
    '''
    Args:

        input (FloatTensor): Any FloatTensor to apply ReLU activation on

        inplace (bool, optional): Will operate inplace if True
    '''
    mask = input < 0

    if inplace:
        return input.masked_fill_(mask, 0)
    return input.clone().masked_fill_(mask, 0)

    ''' Alternative way to threshold

    mask = input >= t
    output = input.clone().fill_(value)

    return output.masked_scatter_(mask, input)
    '''

# ReLU(x)= max(0, x)
class ReLU(Module):
    def _init_(self):
        super(ReLU, self)._init_()

    def forward(self , input):
        self.input = input
        return relu(input, inplace=False)

    def backward(self, grad_wrt_output):
        return (self.input > 0).float() * grad_wrt_output

    def param(self):
        return []


def tanh(input, x):
    pass

class Tanh(Module):
    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *grad_wrt_output):
        raise  NotImplementedError

    def param(self):
        return  []

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Sigmoid(Module):
    def _init_(self):
        super(Sigmoid, self)._init_()

    def forward(self, input):
        self.input = input
        return FloatTensor(list(map(sigmoid, input)))

    def backward(self, grad_wrt_output):
        return FloatTensor(list(map(sigmoid_prime, self.input))) * grad_wrt_output

    def param(self):
        return []
