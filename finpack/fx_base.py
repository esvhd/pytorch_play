import torch
import torch.nn as nn
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


def run_model(model, data):
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
