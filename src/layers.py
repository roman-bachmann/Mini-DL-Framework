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
        out = input.mm(self.w.value.t())
        if self.bias is not None:
            out += self.bias.value
        return out

    def backward(self, grad_wrt_output):
        if self.bias is not None:
            self.bias.grad = grad_wrt_output.sum(0)
        self.w.grad = grad_wrt_output.t().mm(self.input)
        return grad_wrt_output.mm(self.w.value) # Gradient wrt Linear Layer input

    def param(self):
        return [self.w] if self.bias is None else [self.w, self.bias]


class Dropout(Module):
    # dropout proba = p. Other outputs scaled by 1/(1-p) in training. In testing, nothing done
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        if not self.training:
            return input
        self.mask = input.new(input.shape).bernoulli_(1 - self.p)
        return input * self.mask / (1 - self.p)

    def backward(self, grad_wrt_output):
        if not self.training:
            return grad_wrt_output
        return grad_wrt_output * self.mask / (1 - self.p)

    def param(self):
        return []
