import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.preprocessing as skp

import data_util as du
import training


class FXLSTM(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, output_seq_len,
                 bias=True, dropout=0,
                 batch_first=False, ):
        super(FXLSTM, self).__init__()

        assert(num_layers > 0)
        assert(output_seq_len > 0)

        self.output_seq_len = output_seq_len
        self.out_idx = Variable(torch.arange(output_seq_len).long())

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            dropout=dropout, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size * output_seq_len,
                                hidden_size * output_seq_len)

    def forward(self, x):
        z, *_ = self.lstm(x)

        _, N, hidden_size = z.shape

        # extrct the first set of output used
        if z.is_cuda and not self.out_idx.is_cuda:
            self.out_idx = self.out_idx.cuda()
        u = torch.index_select(z, 0, self.out_idx)

        # reshape for linear layer
        a = u.permute([1, 2, 0]).contiguous().view((N, -1))

        z = self.linear(a)
        # change back to LSTM output format
        z = z.view((N, hidden_size, self.output_seq_len)
                   ).permute([2, 0, 1]).contiguous()
        return z


def reshape_rnn(array, inplace=True):
    # pytorch RNN models uses by default (seq_len, batch, input_dim)
    array = array.swapaxes(0, 2)
    array = array.swapaxes(1, 2)
    return array


def run_lstm_model():
        # load data
    print('Load data...')
    train_x, test_x, train_y, test_y = du.load_fx_10m_xy(test_size=.2,
                                                         y_shape_mode=1)

    # normalize train and test X.
    train_x_2d = train_x.swapaxes(1, 2).reshape((-1, train_x.shape[1]))
    test_x_2d = test_x.swapaxes(1, 2).reshape((-1, test_x.shape[1]))

    # normalize
    scaler_x = skp.StandardScaler()

    train_x_2d = scaler_x.fit_transform(train_x_2d)
    # train_y_2d = scaler_x.transform(train_y_2d)

    test_x_2d = scaler_x.transform(test_x_2d)
    # test_y_2d = scaler_x.transform(test_y_2d)

    # reshape to LSTM input shape (seq_len, batch, input_dim)
    print('Swap axes to fit LSTM...')
    train_x = (train_x_2d
               .reshape((-1, train_x.shape[-1], train_x_2d.shape[-1]))
               .swapaxes(1, 2))
    test_x = (test_x_2d
              .reshape((-1, test_x.shape[-1], test_x_2d.shape[-1]))
              .swapaxes(1, 2))

    train_x = reshape_rnn(train_x)
    train_y = reshape_rnn(train_y)
    test_x = reshape_rnn(test_x)
    test_y = reshape_rnn(test_y)

    print('Train X.shape: %s, Train y.shape: %s' %
          (train_x.shape, train_y.shape))
    print('Test X.shape: %s, Test y.shape: %s' %
          (test_x.shape, test_y.shape))

    input_dim = train_x.shape[-1]
    output_seq = train_y.shape[0]

    # hyperparams
    num_layers = 4
    hidden_size = input_dim
    dropout = 0.2

    model = FXLSTM(input_dim, hidden_size, num_layers, output_seq,
                   dropout=dropout)

    # x_train = Variable(torch.from_numpy(train_x).float())
    # y_train = Variable(torch.from_numpy(train_y).float())

    # x_test = Variable(torch.from_numpy(test_x).float())
    # y_test = Variable(torch.from_numpy(test_y).float())

    epochs = 1000
    lr = .01

    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    # loss_func = nn.L1Loss()

    loss = training.run_training(model, (train_x, test_x, train_y, test_y),
                                 loss_func, lr=lr, epochs=epochs,
                                 print_every=100)

    return loss


if __name__ == '__main__':
    run_lstm_model()
