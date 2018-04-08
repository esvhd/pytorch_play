import torch
import torch.nn as nn
# from torch.autograd import Variable
import data_util as du

import fx_base


class FXLSTM(nn.Module):

    def __init__(FXLSTM, self, input_dim, hidden_dim, output_dim, num_layers):
        super(FXLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        # #         dim_x = x.size()
        # # batch_size = x.size()[1]
        # # print(dim_x, batch_size)

        # h0 = torch.randn([self.num_layers, batch_size, self.hidden_dim])
        # c0 = torch.randn([self.num_layers, batch_size, self.hidden_dim])

        y = self.lstm(x)
        return self.linear(y[-1])


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

    # reshape to LSTM input shape (seq_len, batch, input_dim)
    print('Swap axes to fit LSTM...')
    train_x = reshape_rnn(train_x)
    test_x = reshape_rnn(test_x)
    train_y = reshape_rnn(train_y)
    test_y = reshape_rnn(test_y)

    print('Train X.shape: %s, Train y.shape: %s' %
          (train_x.shape, train_y.shape))
    print('Test X.shape: %s, Test y.shape: %s' %
          (test_x.shape, test_y.shape))

    input_dim = train_x.shape[-1]
    output_dim = train_y.shape[-1]

    # hyperparams
    hidden_dim = 128
    num_layers = 2


if __name__ == '__main__':
    run_lstm_model()
