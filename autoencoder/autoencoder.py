# Autoencoder examples
# Auther: ZWL


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dataset
import time
import datetime


class Encoder(nn.Module):

    def __init__(self, input_dim, layer_dims, bias=False,
                 activation='linear',
                 output_activation=None):
        super(Encoder, self).__init__()

        assert(activation in {'linear', 'relu', 'selu', 'tanh', 'sigmoid'})

        self.output_dim = layer_dims[-1]
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.bias = bias

        # determine activation function
        layer_act = choose_activation(activation)
        output_act = choose_activation(output_activation)

        num_layers = len(layer_dims)
        self.model = nn.Sequential()
        for n in range(num_layers):
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

            if n != num_layers - 1:
                if layer_act is not None:
                    self.model.add_module(f'{activation}_{n}', layer_act)
                else:
                    print('No layer activation function.')
            else:
                # output layer
                if output_act is not None:
                    self.model.add_module(f'{activation}_{n}', output_act)
                else:
                    print('No output activation function.')

    def forward(self, x):
        out = self.model.forward(x)
        return out


def choose_activation(activation):
    # determine activation function
    if activation == 'relu':
        act_func = nn.ReLU()
    elif activation == 'selu':
        act_func = nn.SELU()
    elif activation == 'tanh':
        act_func = nn.Tanh()
    elif activation == 'sigmoid':
        act_func = nn.Sigmoid()
    else:
        act_func = None

    return act_func


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, layer_dims, learning_rate, bias=False,
                 activation='linear',
                 encoder_output_activation=None,
                 decoder_output_activation=None):
        '''
        Autoencoder. Decoder is the mirror image of encoder without the
        output activation transform.

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
        activation : str, optional
            Activation function for all layers except the final output layer.
            Default linear, i.e. no activation.
        encoder_output_activation : None, optional
            Activation function for the final output layer for the encoder.
            Default None, same as linear, i.e. no activation.
        decoder_output_activation : None, optional
            Activation function for the final output layer for the decoder.
            Default None, same as linear, i.e. no activation.
        '''
        super(AutoEncoder, self).__init__()

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
            # -2: from element position -2 onwards, i.e. not including the
            # last layer dim from the encoder.
            # `::-1`:reverse order
            de_layer_dims = layer_dims[-2::-1]
            de_layer_dims.append(input_dim)

        # build encoders
        self.encoder = Encoder(input_dim, layer_dims, bias, activation,
                               encoder_output_activation)
        self.decoder = Encoder(de_input_dim, de_layer_dims, bias, activation,
                               decoder_output_activation)

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


def train(model, loss_function, x, learning_rate,
          epochs=1,
          optimizer='adam',
          print_every=100):
    """Train autoencoders.

    Parameters
    ----------
    model : subclasses of torch.nn.Module
        Pytorch model
    loss_function : TYPE
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
            loss = loss_function(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wall_time = time.perf_counter() - wall_time
        proc_time = time.process_time() - proc_time

        if i % print_every == 0:
            print('Epoch: %d, loss=%.5e, perf time=%s, proc time=%s' %
                  (i, loss.data[0],
                   format_seconds(wall_time), format_seconds(proc_time)))

        lost_hist.append(loss.data[0])

    total_wall = time.perf_counter() - total_wall
    total_proc = time.process_time() - total_proc

    print('perf time=%s, process time=%s' %
          (format_seconds(total_wall), format_seconds(total_proc)))

    return lost_hist


if __name__ == '__main__':
    import sklearn.decomposition as skd
    import numpy as np

    # load data
    mnist_train = dataset.MNIST('/home/zwl/data/MNIST', train=True,
                                download=False)

    use_cuda = torch.cuda.is_available()

    x = mnist_train.train_data.float()

    if use_cuda:
        print('Using GPU.')
        x = x.cuda()

    # set up training
    loss_function = torch.nn.MSELoss(size_average=True)
    learning_rate = 3e-4

    # setup architecture, use linear single layer autoencoder
    input_dim = mnist_train.train_data[0].shape[1]
    layer_dims = [10]

    coder = AutoEncoder(input_dim, layer_dims, learning_rate, bias=False,
                        activation='linear',
                        encoder_output_activation=None,
                        decoder_output_activation=None)
    if use_cuda:
        coder = coder.cuda()
    print(coder)

    # training, take first 1k examples
    data_idx = 1000
    train(coder, loss_function, x[:data_idx], learning_rate,
          epochs=100, print_every=20)

    encoding, out = coder.forward(Variable(x[data_idx], requires_grad=False))
    loss = x[data_idx].sub(out.data).pow(2).mean()

    # work out PCA loss
    pca = skd.PCA(n_components=10)
    pca_out = pca.fit_transform(x[data_idx].float())
    inverse = pca.inverse_transform(pca_out)
    pca_loss = np.power((x[data_idx].cpu().numpy() - inverse), 2).mean()

    print(f'Test AE loss: {loss:.5e} vs PCA loss: {pca_loss:.5e}.')
