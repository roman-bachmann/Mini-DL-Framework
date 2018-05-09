import math

def xavier_normal(tensor):
    std_dev = math.sqrt(2 / (tensor.shape[0] + tensor.shape[1]))
    return tensor.normal_(0, std_dev)

def xavier_uniform(tensor):
    std_dev = math.sqrt(2 / (tensor.shape[0] + tensor.shape[1]))
    return tensor.uniform_(-std_dev, std_dev)
