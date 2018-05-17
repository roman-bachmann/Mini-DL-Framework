import math

def xavier_normal(tensor):
    '''
    Returns a Xavier normal initialized tensor using
    standard deviation = sqrt(2 / (N_in + N_out))

    Args:
        tensor (FloatTensor): Tensor to initialize
    '''
    std_dev = math.sqrt(2 / (tensor.shape[0] + tensor.shape[1]))
    return tensor.normal_(0, std_dev)

def xavier_uniform(tensor):
    '''
    Returns a Xavier uniform initialized tensor using
    standard deviation = sqrt(6 / (N_in + N_out))

    Args:
        tensor (FloatTensor): Tensor to initialize
    '''
    std_dev = math.sqrt(6 / (tensor.shape[0] + tensor.shape[1]))
    return tensor.uniform_(-std_dev, std_dev)

def uniform_init(tensor):
    '''
    Returns a uniformly initialized tensor using
    standard deviation = sqrt(3 / N_in)

    Args:
        tensor (FloatTensor): Tensor to initialize
    '''
    std_dev = math.sqrt(3 / tensor.shape[1])
    return tensor.uniform_(-std_dev, std_dev)
