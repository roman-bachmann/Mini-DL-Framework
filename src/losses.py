from module import Module

class LossMSE(Module):
    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def param(self):
        return  []
