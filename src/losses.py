
'''
computes the element-wise square of the difference between `predictions` and `labels`
'''
class LossMSE():
    def __init__(self):
        self.__class__.__call__ = self.apply

    def loss_prime(self, predictions, targets):
        return -2 * (targets.float() - predictions)

    def loss(self, predictions, targets):
        return (predictions - targets.float()).pow(2).sum()

    def apply(self, predictions, targets):
        return self.loss(predictions, targets), self.loss_prime(predictions, targets)

# class CrossEntropyLoss
# class LossMAE
