class Module(object):
    def __init__(self):
        self.training = True

    def forward(self, input):
        # Should get for input, and returns, a tensor or a tuple of tensors.
        raise NotImplementedError

    def backward(self, grad_wrt_output):
        # Should  get  as  input  a  tensor  or  a  tuple  of  tensors  containing  the  gradient  of  the  loss
        # with respect to the module’s output, accumulate the gradient wrt the parameters, and return a
        # tensor or a tuple of tensors containing the gradient of the loss wrt the module’s input.
        raise NotImplementedError

    def param(self):
        # should return a list of pairs, each composed of a parameter tensor, and a gradient tensor
        # of same size.  This list should be empty for parameterless modules (e.g.  ReLU)
        return []

    def zero_grad(self):
        for p in self.param():
            p.grad.zero_()
