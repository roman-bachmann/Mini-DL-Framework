from module import Module

class ReLU(Module):
    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def param(self):
        return  []

class Tanh(Module):
    def forward(self , *input):
        raise  NotImplementedError

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError
        
    def param(self):
        return  []
