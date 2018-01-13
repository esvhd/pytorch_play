# Autoencoder examples
# Auther: ZWL


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dataset
import time
import datetime


class LinearEncoder(nn.Module):

    def __init__(self, input_dim, layer_dims, bias=False):
        super(LinearEncoder, self).__init__()

        self.output_dim = layer_dims[-1]
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.bias = bias

        self.model = nn.Sequential()
        for n in range(len(layer_dims)):
            if n == 0:
                self.model.add_module(f'linear_{n}',
                                      nn.Linear(input_dim,
                                                layer_dims[n],
                                                bias=bias))
            else:
                self.model.add_module(f'linear_{n}',
                                      nn.Linear(layer_dims[n - 1],
                                                layer_dims[n],
                                                bias=bias))

    def forward(self, x):
        out = self.model.forward(x)
        return out


class LinearAutoEncoder(nn.Module):

    def __init__(self, input_dim, layer_dims, learning_rate, bias=False):
        '''
        Linear Autoencoder. Decoder is the mirror image of encoder.

        Parameters
        ----------
        input_dim : int
            input dimension
        layer_dims : list-like
            list of int values for layer dimension
        learning_rate : flat
            learning rate
        bias : bool, optional, default False
            whether to add bias terms
        '''
        super(LinearAutoEncoder, self).__init__()

        self.learning_rate = learning_rate
        self.bias = bias

        # work out decoder dims
        # decoder is the mirror image
        de_input_dim = layer_dims[-1]
        n_layers = len(layer_dims)
        if n_layers < 2:
            de_layer_dims = [input_dim]
        else:
            # indexing:
            # -2 - from element -2 onwards
            # ::-1 - reverse order
            de_layer_dims = layer_dims[-2::-1]
            de_layer_dims.append(input_dim)

        # build encoders
        self.encoder = LinearEncoder(input_dim, layer_dims, bias)
        self.decoder = LinearEncoder(de_input_dim, de_layer_dims, bias)

    def forward(self, x):
        encoding = self.encoder(x)
        out = self.decoder(encoding)
        return encoding, out


def format_seconds(seconds):
    """Format seconds to %H:%M:%S

    Parameters
    ----------
    seconds : float
        seconds

    Returns
    -------
    str
        str format for datetime.timedelta, i.e. %H:%M:%S.
    """
    d = datetime.timedelta(seconds=seconds)
    return str(d)


def train(model, loss_criterion, x,
          learning_rate,
          epochs=1,
          optimizer='adam',
          print_every=100):
    """Train autoencoders.

    Parameters
    ----------
    model : subclasses of torch.nn.Module
        Pytorch model
    loss_criterion : TYPE
        loss function
    x : tensors
        training data
    learning_rate : float
        learning rate
    epochs : int, optional
        default 1.
    optimizer : str, optional
        choice of optimizer, default 'adam'. Others currently not supported.
    print_every : int, optional
        print loss and perf stats for every given epochs.

    Returns
    -------
    list
        training loss history

    Raises
    ------
    NotImplementedError
        Other optimizer methods not yet supported.
    """
    params = model.parameters()

    assert(optimizer in {'adam', 'SGD', 'ada'})

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate)
    else:
        raise NotImplementedError

    lost_hist = []

    # perf time includes sleep and is system wide, seconds.
    total_wall = time.perf_counter()
    # process time does not include sleep time, seconds.
    total_proc = time.process_time()

    for i in range(epochs):
        wall_time = time.perf_counter()
        proc_time = time.process_time()

        for z in x:
            z_x = Variable(z)
            y = Variable(z, requires_grad=False)
            _, y_pred = model(z_x)

            # for auto encoders, x and y are the same.
            loss = loss_criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wall_time = time.perf_counter() - wall_time
        proc_time = time.process_time() - proc_time

        if i % print_every == 0:
            print('Epoch: %d, loss=%.5e, perf time=%s, proc time=%s' %
                  (i,
                   loss.data[0],
                   format_seconds(wall_time),
                   format_seconds(proc_time)))
            # loss.cpu() if self.use_cuda else loss)
        lost_hist.append(loss.data[0])

    total_wall = time.perf_counter() - total_wall
    total_proc = time.process_time() - total_proc

    print('perf time=%s, process time=%s' %
          (format_seconds(total_wall), format_seconds(total_proc)))

    return lost_hist


if __name__ == '__main__':
    # load data
    mnist_train = dataset.MNIST('/home/zwl/data/MNIST', train=True,
                                download=False)
    mnist_test = dataset.MNIST('/home/zwl/data/MNIST', train=False,
                               download=False)

    input_dim = mnist_train.train_data[0].shape[1]
    layer_dims = [28, 10]
    learning_rate = 3e-4

    coder = LinearAutoEncoder(input_dim, layer_dims, learning_rate, bias=False)

    print(coder)

    coder.train(mnist_train.train_data, num_epoch=10)

    y = coder.forward(mnist_test.test_data[3])
    print(y.shape)
