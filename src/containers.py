from module import Module
from losses import LossMSE

class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.module_list = []
        for module in modules:
            self.module_list.append(module)

    def add_module(self, module):
        self.module_list.append(module)

    def train(self):
        for module in self.module_list:
            module.training = True

    def eval(self):
        for module in self.module_list:
            module.training = False

    def forward(self, input):
        for module in self.module_list:
            input = module.forward(input)
        return input

    def backward(self, grad_wrt_output):
        for module in self.module_list[::-1]:
            grad_wrt_output = module.backward(grad_wrt_output)

    def param(self):
        param_list = []
        for module in self.module_list:
            param_list = param_list + module.param()
        return param_list
