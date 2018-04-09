import torch
# import torch.nn as nn
from torch.autograd import Variable

import time


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


def numpy_value(value):
    return value.data.cpu().numpy() if value.is_cuda else value.data.numpy()


def run_training(model, data, loss_func, lr=3e-4, epochs=100, print_every=10,
                 test_loss_func=None):
    # load data
    print('Load data...')
    train_x, test_x, train_y, test_y = data

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

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_func = nn.MSELoss(size_average=True)

    print('\nSet to Training Mode, start training...')
    model.train()

    loss_hist = []
    for i in range(epochs):
        elapsed = time.time()

        loss = train(model, loss_func, opt, x_train, y_train)
        loss_hist.append(numpy_value(loss)[0])

        elapsed = time.time() - elapsed
        if i % print_every == 0:
            if test_loss_func is not None:
                model.eval()
                y_hat = model(x_test)
                test_loss = test_loss_func(y_hat, y_test)
                model.train()

                print('Epoch %d, Time taken (s): %.3f, training loss: %.5f, '
                      'test loss: %.5f' %
                      (i, elapsed, loss, test_loss))
            else:
                print('Epoch %d, Time taken (s): %.3f, training loss: %.5f' %
                      (i, elapsed, loss))

    if test_loss_func is not None:
        model.eval()
        y_hat = model(x_test)
        test_loss = test_loss_func(y_hat, y_test)
        print('\nFinal Test loss: %.5f' % test_loss)

    return (loss_hist, numpy_value(test_loss)[0])
