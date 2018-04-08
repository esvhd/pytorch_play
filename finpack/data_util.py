# import numpy as np
import os.path as osp
import sklearn.model_selection as skms
import h5py


def load_fx_10m_xy(test_size, y_shape_mode=0):
    '''
    Load prepared data.

    Parameters
    ----------
    test_size : TYPE


    Returns
    -------
    train_x, test_x, train_y, test_y
    Shape is (nobs, dims, seq_len)
    '''
    # file has data for first 50 vs last 6 prints
    h5_file = osp.expanduser('~/data/fx/fx_close_10m_xy2.h5')
    with h5py.File(h5_file, mode='r') as store:
        X = store['close_10m_X'][:]
        y = store['close_10m_y'][:]

    X = X.reshape((-1, X.shape[-2], X.shape[-1]))

    # reshape y as last layer of tcn net is a linear layer
    if y_shape_mode == 0:
        # TCN mode
        y = y.reshape((-1, y.shape[-2] * y.shape[-1]))
    else:
        # RNN mode
        y = y.reshape((-1, y.shape[-2], y.shape[-1]))

    xy = skms.train_test_split(X, y, test_size=test_size)

    return xy
