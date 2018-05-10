from torch import FloatTensor, LongTensor
import layers, losses, containers, activations
import math

def generate_disc_set(nb):
    '''Generates random nb*2-dim input data and nb-dim
    output data. y is 1 if x is within sqrt(2/pi) circle, else 0.'''
    x = FloatTensor(nb, 2).uniform_(-1, 1)
    dist = x[:,0] ** 2 + x[:,1] ** 2
    y = [1 if d < 2/math.pi else 0 for d in dist]
    return x, LongTensor(y)

x_train, y_train = generate_disc_set(1000)
x_test, y_test = generate_disc_set(1000)

# Normalise data
mu, std = x_train.mean(), x_train.std()
x_train.sub_(mu).div_(std)
x_test.sub_(mu).div_(std)

# Check if there are approximately 50% of each class
y_train.sum(), y_test.sum()

def convert_to_one_hot_labels(input, target):
    '''Convert output to one-hot labled tensor'''
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

y_train = convert_to_one_hot_labels(x_train, y_train)
y_test = convert_to_one_hot_labels(x_train, y_test)

model = containers.Sequential(
            layers.Linear(2, 128, with_bias=True),
            activations.ReLU(),
            layers.Linear(128, 2, with_bias=True),
            activations.Sigmoid()
)

criterion = losses.LossMSE()

def compute_nb_errors(model, data_input, data_target):
    n_misclassified = 0
    for input, target in zip(data_input, data_target):
        output = model.forward(input)
        output_class = output.max(0)[1][0]
        target_class = target.max(0)[1][0]
        if output_class != target_class:
            n_misclassified = n_misclassified + 1
    error = n_misclassified / len(data_input)
    return error

def train_model(model, train_input, train_target, n_epochs=10, eta=0.1, verbose=0):
    for e in range(n_epochs):
        sum_loss = 0

        for x, y in zip(train_input, train_target):
            output = model.forward(x)
            loss, grad_wrt_output = criterion(output, y)
            sum_loss += loss
            model.zero_grad()
            model.backward(grad_wrt_output)
            for p in model.param():
                p.value = p.value - eta * p.grad
        if verbose:
            print(e, sum_loss)

train_model(model, x_train, y_train, n_epochs=250, eta=0.1, verbose=1)
error = compute_nb_errors(model, x_test, y_test)
print('Test error: {:.2f}%'.format(error*100))
