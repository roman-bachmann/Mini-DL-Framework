class Parameter():
    '''
    Holds a parameter and its gradient for ease of access outside a module.

    Args:
        value (FloatTensor): Parameter Tensor (can be a weight, bias, etc.)

        grad (FloatTensor): Gradient Tensor of parameter
    '''
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad
