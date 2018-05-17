class LossMSE():
    '''
    Mean Squared Error loss function
    '''
    def __init__(self):
        self.__class__.__call__ = self.apply

    def loss_prime(self, predictions, targets):
        '''
        Returns the derivative of the MSE loss.

        Args:
            predictions (FloatTensor): Neural network output

            targets (LongTensor): True target vector
        '''
        return -2 * (targets.float() - predictions)

    def loss(self, predictions, targets):
        '''
        Returns the MSE loss.

        Args:
            predictions (FloatTensor): Neural network output

            targets (LongTensor): True target vector
        '''
        return (predictions - targets.float()).pow(2).sum(1).mean()

    def apply(self, predictions, targets):
        '''
        Returns the loss and the derivative of the loss.

        Args:
            predictions (FloatTensor): Neural network output

            targets (LongTensor): True target vector
        '''
        return self.loss(predictions, targets), self.loss_prime(predictions, targets)
