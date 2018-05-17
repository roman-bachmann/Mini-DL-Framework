from module import Module
from losses import LossMSE

class Sequential(Module):
    '''
    Sequential container holding a list of neural network layers and activations.
    Modules can all be passed by argument when initializing the container. For example:

        model = Sequential(
            layers.Linear(28,64),
            layers.ReLU(),
            layers.Dropout(),
            layers.Linear(64,2)
        )

    When using training / evaluation specific Modules (like Dropout), container can be
    set to train or eval modes.

    Forward and backward passes can be called and do not have to be implemented oneself.

    Args:
        *modules (Module): Several neural network layers and activations
    '''
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.module_list = []
        for module in modules:
            self.module_list.append(module)

    def add_module(self, module):
        '''
        Adds a module to the back of the module list.

        Args:
            module (Module): The module to append at the back of the sequential stack
        '''
        self.module_list.append(module)

    def train(self):
        ''' Set all modules to training mode. '''
        for module in self.module_list:
            module.training = True

    def eval(self):
        ''' Set all modules to evaluation mode. '''
        for module in self.module_list:
            module.training = False

    def forward(self, input):
        '''
        Compute the forward pass over all modules sequentially.

        Args:
            input (FloatTensor): Input Tensor to neural network
        '''
        for module in self.module_list:
            input = module.forward(input)
        return input

    def backward(self, grad_wrt_output):
        '''
        Compute the backward pass over all modules sequentially in an inverse manner.

        Args:
            grad_wrt_output (FloatTensor): Gradient of loss with respect to network output
        '''
        for module in self.module_list[::-1]:
            grad_wrt_output = module.backward(grad_wrt_output)

    def param(self):
        ''' Return list of all parameters of all modules. '''
        param_list = []
        for module in self.module_list:
            param_list = param_list + module.param()
        return param_list
