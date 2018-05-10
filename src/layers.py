from module import Module
from parameter import Parameter
from initializers import xavier_normal
from torch import FloatTensor

class Linear(Module):

    def __init__(self, in_dim, out_dim, with_bias=True):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        w = FloatTensor(out_dim, in_dim)
        w = xavier_normal(w)
        w_grad = FloatTensor(out_dim, in_dim).zero_()
        self.w = Parameter(w, w_grad)

        self.bias = None
        if with_bias:
            bias = FloatTensor(out_dim).zero_()
            bias_grad = FloatTensor(out_dim).zero_()
            self.bias = Parameter(bias, bias_grad)

    def forward(self, input):
        self.input = input
        out = self.w.value.mv(input)
        if self.bias is not None:
            out += self.bias.value
        return out

    def backward(self, grad_wrt_output):
        if self.bias is not None:
            self.bias.grad = grad_wrt_output
        self.w.grad = grad_wrt_output.view(-1,1).mm(self.input.view(1,-1))
        return self.w.value.t().mv(grad_wrt_output) # Gradient wrt Linear Layer input

    def param(self):
        return [self.w] if self.bias is None else [self.w, self.bias]