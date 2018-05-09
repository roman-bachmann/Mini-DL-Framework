from module import Module
from initializers import xavier_normal
from torch import FloatTensor

class Linear(Module):

    def __init__(self, in_dim, out_dim, bias=True):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = FloatTensor(out_dim, in_dim)
        self.w = xavier_normal(self.w)
        self.w_grad = FloatTensor(out_dim, in_dim).zero_()

        if bias:
            self.bias = FloatTensor(out_dim, 1).zero_()
            self.bias_grad = FloatTensor(out_dim, 1).zero_()
        else:
            self.bias = None
            self.bias_grad = None

    def forward(self, input):
        out = self.w.mm(input.view(len(input), 1))
        if self.bias is not None:
            out += self.bias
        return out

    def param(self):
        return [[self.w, self.w_grad],
                [self.bias, self.bias_grad]]
