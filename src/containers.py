from module import Module

class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.module_list = []
        for module in modules:
            self.module_list.append(module)

    def add_module(self, module):
        self.module_list.append(module)

    def forward(self , *input):
        for module in module_list:
            input = module.forward(input)
        return input

    def backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def param(self):
        return  []
