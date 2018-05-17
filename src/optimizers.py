import math

class RMSProp():
    '''
    RMSProp (Root Mean Squared Propagation) optimizer as proposed by Geoffrey Hinton in
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    Keeps a moving average of the squared gradient for each weight.

    Args:
        params (List[Parameter]): List of Parameters to optimize

        learning_rate (float, optional): Learning rate for weight update

        decay_rate (float, optional): Decay rate of the squared gradient term

        eps (float, optional): Added for numerical stability when dividing by squared gradient
    '''
    def __init__(self, params, learning_rate=0.001, decay_rate=0.9, eps=1e-6):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.eps = eps
        # Initializing a list of parameters with their mean squared gradient
        self.plist = [{'param': p,
                       'mean_square': p.value.new(p.value.shape).zero_()}
                      for p in params]

    def step(self):
        ''' Performs the optimization step and updates all parameters accordingly. '''
        for p in self.plist:
            param = p['param']
            # Update second moment estimate
            p['mean_square'] = self.decay_rate * p['mean_square'] + \
                               (1 - self.decay_rate) * param.grad * param.grad
            # Compute and apply update
            delta = - self.learning_rate * param.grad / (p['mean_square'] + self.eps).sqrt()
            param.value += delta

class Adam():
    '''
    Adam (Adaptive Moment Estimation) optimizer, adaptively computing learning rates
    for each weight.

    Args:
        params (List[Parameter]): List of Parameters to optimize

        learning_rate (float, optional): Learning rate for weight update

        p1 (float, optional): Exponential decay rate for first moment

        p2 (float, optional): Exponential decay rate for second moment

        eps (float, optional): Added for numerical stability when dividing by squared gradient
    '''
    def __init__(self, params, learning_rate=0.001, p1=0.9, p2=0.999, eta=1e-8):
        self.params = params
        self.learning_rate = learning_rate
        self.p1 = p1
        self.p2 = p2
        self.eta = eta
        self.t = 0
        # Initializing a list of parameters with their first and second moment estimates
        self.plist = [{'param': p,
                       'm1': p.value.new(p.value.shape).zero_(),
                       'm2': p.value.new(p.value.shape).zero_()}
                      for p in params]


    def step(self):
        ''' Performs the optimization step and updates all parameters accordingly. '''
        self.t += 1
        for p in self.plist:
            param = p['param']
            # Update biased first moment estimate
            p['m1'] = self.p1 * p['m1'] + (1 - self.p1) * param.grad
            # Update biased second moment estimate
            p['m2'] = self.p2 * p['m2'] + (1 - self.p2) * param.grad * param.grad
            # Correct bias in first moment
            m1_corr = p['m1'] / (1 - math.pow(self.p1, self.t))
            # Correct bias in second moment
            m2_corr = p['m2'] / (1 - math.pow(self.p2, self.t))
            # Compute and apply update
            delta = - self.learning_rate * m1_corr / (m2_corr.sqrt() + self.eta)
            param.value += delta
