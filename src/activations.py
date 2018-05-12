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


class Tanh(Module):
    def _init_(self):
        super(Tanh, self)._init_()

    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self, grad_wrt_output):
        return (1 - self.input.tanh().pow(2)) * grad_wrt_output

    def param(self):
        return []


class Sigmoid(Module):
    def _init_(self):
        super(Sigmoid, self)._init_()

    def forward(self, input):
        self.input = input
        return 1 / (1 + (-input).exp())

    def backward(self, grad_wrt_output):
        s = 1 / (1 + (-self.input).exp())
        return (s * (1 - s)) * grad_wrt_output

    def param(self):
        return []


# class Softmax(Module):
#     def _init_(self):
#         super(Sigmoid, self)._init_()
#
#     def forward(self, input):
#         self.input = input
#         exp_input = (input - input.max()).exp()
#         return exp_input / exp_input.sum()
#
#     def backward(self, grad_wrt_output):
#         return FloatTensor(list(map(softmax_prime, self.input))) * grad_wrt_output
#
#     def param(self):
#         return []

# class LeakyRelu...
