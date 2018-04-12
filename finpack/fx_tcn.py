import torch
import torch.nn as nn
# from torch.autograd import Variable

import training
import data_util as du

# import time
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

    epochs = 500
    lr = .001
    loss_func = nn.L1Loss(size_average=True)

    loss = training.run_training(model, (train_x, test_x, train_y, test_y),
                                 loss_func, lr=lr, epochs=epochs,
                                 print_every=100,
                                 test_loss_func=torch.nn.functional.l1_loss)

    return loss


if __name__ == '__main__':
    run_model()
