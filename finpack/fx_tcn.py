import torch
import torch.nn as nn
from torch.autograd import Variable
import data_util as du

import time
import TCN.tcn as tcn


class FXTCN(nn.Module):

    def __init__(self, input_channels, output_size, channel_sizes,
                 kernel_size=2,
                 dropout=0.2):
        super(FXTCN, self).__init__()
        self.tcn = tcn.TemporalConvNet(input_channels, channel_sizes,
                                       kernel_size=kernel_size,
                                       dropout=dropout)
#         self.linear = nn.Linear(channel_sizes[-1], output_size)
        self.linear = nn.Linear(channel_sizes[-1], output_size)

    def forward(self, x):
        # x.shape == (N, channels, dim)
        y = self.tcn(x)
        # only use the last entry slice of y
        y = self.linear(y[:, :, -1])
        return y.contiguous()


def train(model, loss, optimizer, x, y):
    # Reset gradient
    optimizer.zero_grad()

    # Forward
#     print(x)
    fx = model.forward(x)
#     print(fx)
    output = loss(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output


def run_model():
    # load data
    print('Load data...')
    train_x, test_x, train_y, test_y = du.load_fx_10m_xy(test_size=.2,
                                                         y_shape_mode=0)

    seq_len = train_x.shape[-1]
    input_channels = train_x.shape[-2]
    output_size = train_y.shape[-1]

    print('seq_len: %d, input_channels: %d, output_size: %d' %
          (seq_len, input_channels, output_size))

    # hyperparameters
    kernel_size = 3
    dropout = .2
    channel_sizes = [128, 128]

    model = FXTCN(input_channels, output_size,
                  channel_sizes=channel_sizes,
                  kernel_size=kernel_size,
                  dropout=dropout)

    if torch.cuda.is_available():
        print('Using GPU.')
        x_train = Variable(torch.from_numpy(train_x).float()).cuda()
        y_train = Variable(torch.from_numpy(train_y).float()).cuda()

        x_test = Variable(torch.from_numpy(test_x).float()).cuda()
        y_test = Variable(torch.from_numpy(test_y).float()).cuda()

        model = model.cuda()

    print('Train X.shape: %s, Train y.shape: %s' %
          (train_x.shape, train_y.shape))
    print('Test X.shape: %s, Test y.shape: %s' %
          (test_x.shape, test_y.shape))

    epochs = 100
    lr = .001
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss(size_average=True)

    print('\nTraining...')
    model.train()
    loss_hist = []
    for i in range(epochs):
        elapsed = time.time()

        loss = train(model, loss_func, opt, x_train, y_train)
        loss_hist.append(loss)

        elapsed = time.time() - elapsed
        if i % 10 == 0:
            print('Epoch %d, Time taken (s): %.3f, loss: %.5f' %
                  (i, elapsed, loss))

    model.eval()
    y_hat = model(x_test)
    test_loss = torch.nn.functional.mse_loss(y_hat, y_test, size_average=True)
    print('\nTest loss: %.5f' % test_loss)

    return (loss_hist, test_loss)


if __name__ == '__main__':
    run_model()
