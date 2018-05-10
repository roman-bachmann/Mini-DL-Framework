from torch import FloatTensor, LongTensor
import layers, losses, containers, activations
import math

def generate_disc_set(nb):
    '''Generates random nb*2-dim input data and nb-dim
    output data. y is 1 if x is within sqrt(2/pi) circle, else 0.'''
    x = FloatTensor(nb, 2).uniform_(0, 1)
    squared_dist = x[:,0] ** 2 + x[:,1] ** 2
    y = [1 if d < 1/(2*math.pi) else 0 for d in squared_dist]
    return x, LongTensor(y)

x_train, y_train = generate_disc_set(1000)
x_test, y_test = generate_disc_set(1000)

def convert_to_one_hot_labels(input, target, zero_value=0):
    '''Convert output to one-hot labeled tensor. Value at label position will be 1
    and zero_value everywhere else.'''
    tmp = input.new(target.size(0), target.max() + 1).fill_(zero_value)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

y_train = convert_to_one_hot_labels(x_train, y_train, -1)
y_test = convert_to_one_hot_labels(x_train, y_test, -1)

model = containers.Sequential(
            layers.Linear(2, 25, with_bias=True),
            activations.ReLU(),
            layers.Linear(25, 25, with_bias=True),
            activations.ReLU(),
            layers.Linear(25, 25, with_bias=True),
            activations.ReLU(),
            layers.Linear(25, 2, with_bias=True),
            activations.Tanh()
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

train_model(model, x_train, y_train, n_epochs=150, eta=0.005, verbose=1)

error = compute_nb_errors(model, x_train, y_train)
print('Train error: {:.2f}%'.format(error*100))

error = compute_nb_errors(model, x_test, y_test)
print('Test error: {:.2f}%'.format(error*100))
