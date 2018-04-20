from module import Module
from torch import FloatTensor

class Linear(Module):

    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = FloatTensor(out_dim, in_dim)
        # TODO: Initialize weights

    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def param(self):
        return  []
