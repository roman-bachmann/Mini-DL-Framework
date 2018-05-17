class Module(object):
    '''
    Abstract Module class specifying the functions all modules should implement.
    '''
    def __init__(self):
        # Flag needed for layers that are training / evaluating dependent
        self.training = True

    def forward(self, input):
        '''
        Computes the forward pass in any module. Returns a FloatTensor.

        Args:
            input (FloatTensor): Input Tensor for which forward pass should be calculated
        '''
        raise NotImplementedError

    def backward(self, grad_wrt_output):
        '''
        Computes the backward pass in any module. If needed, computes and accumulates
        the gradient of the loss with respect to the module parameters inside the module.

        Returns the gradient of the loss with respect to the module input.

        Args:
            grad_wrt_output (FloatTensor): Gradient of loss with respect to module output
        '''
        raise NotImplementedError

    def param(self):
        '''
        Returns a list of all Parameters of the module. This list should be empty for
        parameterless modules (e.g. ReLU).
        '''
        return []

    def zero_grad(self):
        '''
        Sets all Parameter gradients to zero.
        '''
        for p in self.param():
            p.grad.zero_()
