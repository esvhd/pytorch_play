import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fx_lstm import load_data


class FXMean(nn.Module):

    def __init__(self, output_len):
        '''
        Navie prediction class, always predict mean of input sequence.

        Parameters
        ----------
        output_len : TYPE
            output sequence length
        '''
        super(FXMean, self).__init__()

        self.output_len = output_len

    def forward(self, x):
        '''
        Return mean of the last dimension.

        Parameters
        ----------
        x : TYPE
            Shape (N, D, L), N: batch size, D: dimension, L: sequence length
        '''
        mean = torch.mean(x, dim=0, keepdim=True)
        _, N, D = mean.shape
        # print(mean.shape)
        y = mean.expand([self.output_len, N, D])
        return y


class FXLast(nn.Module):

    def __init__(self, output_len):
        '''
        Navie prediction class, always predict the same as the last element
        in input sequence.

        Parameters
        ----------
        output_len : TYPE
            output sequence length
        '''
        super(FXLast, self).__init__()
        self.output_len = output_len

    def forward(self, x):
        '''
        Return mean of the last dimension.

        Parameters
        ----------
        x : TYPE
            Shape (N, D, L), N: batch size, D: dimension, L: sequence length
        '''
        L, N, D = x.shape
        # print(mean.shape)
        idx = Variable(torch.LongTensor([L - 1]), requires_grad=False)
        if x.is_cuda:
            idx = idx.cuda()
        y = torch.index_select(x, 0, idx)
        y = y.expand([self.output_len, N, D])
        return y


def run_naive():
    # load data
    print('Load data...')
    train_x, test_x, train_y, test_y = load_data(.2)

    print('Train X.shape: %s, Train y.shape: %s' %
          (train_x.shape, train_y.shape))
    print('Test X.shape: %s, Test y.shape: %s' %
          (test_x.shape, test_y.shape))

    print('Running Naive Mean Model...')

    x_train = Variable(torch.from_numpy(train_x).float(), requires_grad=False)
    x_test = Variable(torch.from_numpy(test_x).float(), requires_grad=False)
    y_train = Variable(torch.from_numpy(train_y).float(), requires_grad=False)
    y_test = Variable(torch.from_numpy(test_y).float(), requires_grad=False)

    output_len = train_y.shape[0]
    model = FXMean(output_len)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        x_test = x_test.cuda()
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        model = model.cuda()

    # there is no trainable parameters, so only need to run in forward model.
    model.eval()

    y_hat = model.forward(x_train)
    train_loss = F.l1_loss(y_hat, y_train)

    y_hat = model.forward(x_test)
    test_loss = F.l1_loss(y_hat, y_test)

    print('Navie Mean Model: Train loss: %.5f, Test loss: %.5f' %
          (train_loss, test_loss))

    # FXLast model
    print('Running Navie Last Model...')
    model = FXLast(output_len)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    y_hat = model.forward(x_train)
    train_loss = F.l1_loss(y_hat, y_train)

    y_hat = model.forward(x_test)
    test_loss = F.l1_loss(y_hat, y_test)

    print('Navie Last Model: Train loss: %.5f, Test loss: %.5f' %
          (train_loss, test_loss))


if __name__ == '__main__':
    run_naive()
