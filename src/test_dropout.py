from torch import FloatTensor, LongTensor
import layers, losses, containers, activations, optimizers
import math
import random
import matplotlib.pyplot as plt

def generate_disc_set(nb, sigma=0):
    '''Generates random nb*2-dim input data and nb-dim
    output data. y is 1 if x is within sqrt(2/pi) circle, else 0.
    Adds some gaussian noise to the circle.'''
    x = FloatTensor(nb, 2).uniform_(0, 1)
    squared_dist = x[:,0] ** 2 + x[:,1] ** 2
    y = [1 if d < 1/(2*math.pi) + random.gauss(0,sigma) else 0 for d in squared_dist]
    return x, LongTensor(y)

x_train, y_train = generate_disc_set(1000, sigma=0.25)
x_test, y_test = generate_disc_set(1000, sigma=0)

def convert_to_one_hot_labels(input, target, zero_value=0):
    '''Convert output to one-hot labeled tensor. Value at label position will be 1
    and zero_value everywhere else.'''
    tmp = input.new(target.size(0), target.max() + 1).fill_(zero_value)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

y_train = convert_to_one_hot_labels(x_train, y_train, -1)
y_test = convert_to_one_hot_labels(x_train, y_test, -1)

input_sets = {'train': x_train, 'val': x_test}
target_sets = {'train': y_train, 'val': y_test}

n_epochs = 1000

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

def train_model(model, input_sets, target_sets, optimizer, n_epochs=10, batch_size=100, verbose=0):
    losses = {'train': [], 'val': []}
    for e in range(n_epochs):
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
            else:
                model.eval()
            sum_loss = 0
            for b in range(0, input_sets[mode].size(0), batch_size):
                output = model.forward(input_sets[mode].narrow(0, b, batch_size))
                loss, grad_wrt_output = criterion(output, target_sets[mode].narrow(0, b, batch_size))
                sum_loss = sum_loss + loss
                if mode == 'train':
                    model.zero_grad()
                    model.backward(grad_wrt_output)
                    optimizer.step()
            losses[mode].append(sum_loss)
        if verbose:
            print('Epoch {}: Train loss = {:.6f}, val loss = {:.6f}'.format(e, losses['train'][-1], losses['val'][-1]))
    return losses


### Testing without Dropout ###

# Defining the model architecture
model = containers.Sequential(
            layers.Linear(2, 500, with_bias=True),
            activations.ReLU(),
            layers.Linear(500, 500, with_bias=True),
            activations.ReLU(),
            layers.Linear(500, 500, with_bias=True),
            activations.ReLU(),
            layers.Linear(500, 500, with_bias=True),
            activations.ReLU(),
            layers.Linear(500, 2, with_bias=True),
            activations.Tanh()
)

criterion = losses.LossMSE()
optimizer = optimizers.Adam(model.param(), learning_rate=0.001, p1=0.9, p2=0.999)



losses_dict = train_model(model, input_sets, target_sets, optimizer, n_epochs=n_epochs, batch_size=100, verbose=1)

model.eval()
error = compute_nb_errors(model, x_train, y_train)
print('\nTrain error: {:.2f}%'.format(error*100))

error = compute_nb_errors(model, x_test, y_test)
print('Test error: {:.2f}%'.format(error*100))

plt.figure()
plt.plot(list(range(n_epochs)), losses_dict['train'])
plt.plot(list(range(n_epochs)), losses_dict['val'])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


### Testing with Dropout ###

# Defining the model architecture
model = containers.Sequential(
            layers.Linear(2, 500, with_bias=True),
            activations.ReLU(),
            layers.Dropout(0.5),
            layers.Linear(500, 500, with_bias=True),
            activations.ReLU(),
            layers.Dropout(0.5),
            layers.Linear(500, 500, with_bias=True),
            activations.ReLU(),
            layers.Dropout(0.5),
            layers.Linear(500, 500, with_bias=True),
            activations.ReLU(),
            layers.Dropout(0.5),
            layers.Linear(500, 2, with_bias=True),
            activations.Tanh()
)

criterion = losses.LossMSE()
optimizer = optimizers.Adam(model.param(), learning_rate=0.001, p1=0.9, p2=0.999)

losses_dict = train_model(model, input_sets, target_sets, optimizer, n_epochs=n_epochs, batch_size=100, verbose=1)

model.eval()
error = compute_nb_errors(model, x_train, y_train)
print('\nTrain error: {:.2f}%'.format(error*100))

error = compute_nb_errors(model, x_test, y_test)
print('Test error: {:.2f}%'.format(error*100))

plt.figure()
plt.plot(list(range(n_epochs)), losses_dict['train'])
plt.plot(list(range(n_epochs)), losses_dict['val'])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
