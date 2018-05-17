from torch import FloatTensor, LongTensor
import layers, losses, containers, activations, optimizers
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

# Defining the model architecture
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
optimizer = optimizers.Adam(model.param(), learning_rate=0.001, p1=0.9, p2=0.999)

def compute_nb_errors(model, data_input, data_target):
    mini_batch_size = 100
    n_misclassified = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        batch_output = model.forward(data_input.narrow(0, b, mini_batch_size))
        batch_target = data_target.narrow(0, b, mini_batch_size)
        output_class = batch_output.max(1)[1]
        target_class = batch_target.max(1)[1]
        n_misclassified += (output_class != target_class).sum()
    error = n_misclassified / data_input.size(0)
    return error

def train_model(model, train_input, train_target, optimizer, n_epochs=10, batch_size=100, verbose=0):
    for e in range(n_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            output = model.forward(train_input.narrow(0, b, batch_size))
            loss, grad_wrt_output = criterion(output, train_target.narrow(0, b, batch_size))
            sum_loss = sum_loss + loss
            model.zero_grad()
            model.backward(grad_wrt_output)
            optimizer.step()
        if verbose:
            print(e, sum_loss)

model.train()
train_model(model, x_train, y_train, optimizer, n_epochs=300, batch_size=100, verbose=1)

model.eval()
error = compute_nb_errors(model, x_train, y_train)
print('Train error: {:.2f}%'.format(error*100))

error = compute_nb_errors(model, x_test, y_test)
print('Test error: {:.2f}%'.format(error*100))
