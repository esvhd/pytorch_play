import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.keras as K

import pandas as pd
import numpy as np
import scipy.stats as ss
# import matplotlib.pyplot as plt


def MultivariateGaussianTril(mu, scale):
    '''
    Create multivariate Gaussian distribution from mu and lower trianglar
    covariance matrix.

    Parameters
    ----------
    mu : list
        mean parameters
    scale : list
        scale of multivariate gaussian. covariance = scale @ scale.T

    Returns
    -------
    Multivariate Gaussian distribution
    '''
    scale_tril = tfd.fill_triangular(scale, upper=False)
    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
    return mvn


class MixtureDensityNet:

    def __init__(self, num_mixtures, input_dims):
        self.num_mixtures = num_mixtures
        self.input_dims = input_dims

        self.mu_dims = input_dims * num_mixtures
        self.sigma_dims = len(np.tril_indices(input_dims)[0]) * num_mixtures

    def mdn_loss(self, y_true, y_pred, debug=False):
        '''
        For each sample, there will be num_mixtures of multivariate gaussian
        distributions needed.

        In batch mode, this is looking at batch size of examples, for each
        example, there is a set of (alpha, mu, sigma) output from the network.

        y_pred:
            Dimension: (batch_size, network_output_size).
        '''
        idx = self.num_mixtures

        # separate alpha, mu, and sigma
        # first section is alpha, i.e. weights
        alphas = y_pred[:, :idx]
        # need to work on this assert here
    #     alpha_check = tf.reduce_sum(alphas, axis=-1).eval()
    #     assert(np.allclose(alpha_check, 1.))

        mu = y_pred[:, idx:idx + self.mu_dims]

        idx += self.mu_dims
        sigma = y_pred[:, idx:]

        # now for each mixture, construction distribution
        # for each mixture, the no. of mu's == input_dims
        mu_len = self.input_dims
        sigma_len = self.sigma_dims // self.num_mixtures

        if debug:
            print('y_true.shape: ', y_true.shape)
            print('alphas.shape: ', alphas.shape)
            print('mu.shape: ', mu.shape)
            print('sigma.shape: ', sigma.shape)

        # pair up mu's and covariances starting indices
        # indices are in chucks due to the size of mu and sigma arrays.
        # the number of pairs match the number of mixtures
        mu_idx = [x for x in range(0, self.mu_dims, mu_len)]
        sigma_idx = [x for x in range(0, self.sigma_dims, sigma_len)]

        assert(len(mu_idx) == self.num_mixtures)
        assert(len(sigma_idx) == self.num_mixtures)

        if debug:
            # loop, so can print out results
            probs = []
            for i, j in zip(mu_idx, sigma_idx):
                mux = mu[:, i:i + mu_len]
                sigmax = sigma[:, j:j + sigma_len]
                print('mux.shape: ', mux.shape)
                print('sigmax.shape: ', sigmax.shape)
                mvn = MultivariateGaussianTril(mux, sigmax)
                prob = mvn.prob(y_true)
                probs.append(prob)
        else:
            # compute pdf for the batch, for each mixture
            probs = [MultivariateGaussianTril(mu[:, i:i + mu_len],
                                              sigma[:, j:j + sigma_len])
                     .prob(y_true)
                     for i, j in zip(mu_idx, sigma_idx)]

        # shape: (num_features * len(y_true))
        probs = tf.concat(probs, axis=0)
        # reshape back to apply alpha weights
        probs = tf.reshape(probs, (-1, self.num_mixtures))

        result = tf.multiply(probs, alphas)
        if debug:
            print('after product: ', result.shape)

        result = tf.reduce_sum(result, keepdims=True)
        if debug:
            print('after sum: ', result.shape)
        result = -tf.log(result)

        return tf.reduce_mean(result)

    def make_model(self):
        inputs = K.layers.Input(shape=(self.input_dims,), dtype='float32')

        alpha_out = K.layers.Dense(self.num_mixtures,
                                   activation='sigmoid',
                                   name='alpha')(inputs)
        alpha_out = K.layers.Activation('softmax',
                                        name='alpha_softmax')(alpha_out)

        mu_out = K.layers.Dense(self.mu_dims,
                                activation=None,
                                name='mu')(inputs)

        sigma_out = K.layers.Dense(self.sigma_dims,
                                   activation='tanh',
                                   name='sigma_tanh')(inputs)
        sigma_out = K.layers.Dense(self.sigma_dims, activation=None,
                                   name='sigma_linear')(sigma_out)

        outputs = K.layers.concatenate([alpha_out, mu_out, sigma_out], axis=-1)

        # input is output - trying to recover itself
        # loss_func = mdn_loss(alpha_out, mu_out, sigma_out, inputs)

        model = K.models.Model(inputs, outputs)

        model.compile('adam', loss=self.mdn_loss)

        self.model = model


def split_mdn_output(y_hat, num_mixtures, mu_dims, scale_dims):
    '''
    Split alpha, mu and scale parameters frmo the output of a mixture
    density network

    Parameters
    ----------
    y_hat : TYPE
        output from mixture density nets
    num_mixtures : TYPE
        number of mixtures
    mu_dims : TYPE
        length of mu parameters, should be num_mixtures * single density mu
        size.
    scale_dims : TYPE
        length of scale parameters, should be num_mixtues * single density
        scale size.

    Returns
    -------
    tuple of (alpha, mu, scale)
    '''
    N, y_dim = y_hat.shape
    total = num_mixtures + mu_dims + scale_dims
    assert(total == y_dim), print(f'{total} != {y_dim}')

    alpha = y_hat[:, :num_mixtures]

    idx = num_mixtures + mu_dims
    mu = y_hat[:, num_mixtures:idx]

    scale = y_hat[:, idx:]

    # reshape
    mu = mu.reshape((N, num_mixtures, -1))
    scale = scale.reshape((N, num_mixtures, -1))

    return (alpha, mu, scale)


def symmetric_from_tril(N: int, lower):
    '''
    Popular full symmetric matrix from lower triangular matrix.

    Note that numpy populates lower triangular matrxi from top to bottom,
    unlike tensorflow's fill_triangular().

    Parameters
    ----------
    N : int
        matrix dimension
    lower : array
        lower triangular values

    Returns
    -------
    Symmetric matrix
    '''
    x = np.zeros((N, N))
    x[np.tril_indices(N)] = lower
    y = x + x.T
    y[np.diag_indices(N)] -= np.diag(x)
    return y


def generate_data(nobs):
    # these are the parameters we are trying to recover
    N = 3
    cov = [[.3, .2], [.2, .3]]
    z = [([0.5, .3], cov), ([0.3, 0.0], cov), ([0., .1], cov)]

    # sample from 3 bi-variate distributions
    samples = [ss.multivariate_normal(mean=u, cov=v).rvs(size=nobs)
               for u, v in z]
    samples = np.array(samples, dtype='float32')

    weights = np.array([.3, .6, .1], dtype='float32').reshape((3, 1))

    # mix it
    data = np.multiply(weights, samples.reshape(N, -1))
    data = data.reshape(N, nobs, -1)

    x_train = data.sum(axis=0)
    y_train = x_train

    return x_train, y_train


def sample_gaussian_tril(mu, scale, n: int=1):
    '''
    Sample from Multivariate Normal distribution.

    Parameters
    ----------
    mu : TYPE
        mean
    scale : TYPE
        lower triangular for scale
    n : int, optional
        number of samples, default 1.

    Returns
    -------
    Samples
    '''
    mvn = MultivariateGaussianTril(mu, scale)
    samples = mvn.sample(n).eval()
    # with tf.Session() as sess:
    #    samples = mvn.sample(n).eval()
    return samples


# def sample_gaussian_numpy(mu, scale, n: int=1):
#     lower = tfd.fill_triangular(scale, upper=False).eval()
#     cov = np.dot(lower, lower.T)
#     samples = np.random.multivariate_normal(mu, cov, n)
#     return samples


def load_fin_data():
    etf = pd.read_hdf('/home/zwl/data/etf.h5', key='close')
    cols = ['SPY US Equity', 'HYG US Equity', 'IEF US Equity']
    px = etf[cols]
    rtns = np.log(px) - np.log(px.shift(1))
    rtns = rtns.dropna(how='any')
    return rtns


if __name__ == '__main__':
    import time
    import os.path as osp
    import mixture_density_nets.MDN as M
    # # x_train, y_train = generate_data(5000)

    # x_train = y_train = load_fin_data()
    # x_train *= 1000

    # num_mixtures = 5
    # _, mu_dims = x_train.shape
    # scale_dims, *_ = np.tril_indices(mu_dims)[0].shape

    # mdn = MixtureDensityNet(num_mixtures, mu_dims)
    # mdn.make_model()

    # print(mdn.model.summary())

    # print('\nTraining...')
    # epochs = 10
    # history = mdn.model.fit(x_train, y_train, epochs=epochs)

    # output = mdn.model.predict(x_train)
    # alpha, mu, scale = split_mdn_output(output, num_mixtures,
    #                                     mu_dims * num_mixtures,
    #                                     scale_dims * num_mixtures)
    # # make sure weights sum to 1
    # assert(np.allclose(alpha.sum(axis=1), 1.))

    # # save model
    # np.save(osp.expanduser('~/tmp/alpha.npy'), alpha)
    # np.save(osp.expanduser('~/tmp/mu.npy'), mu)
    # np.save(osp.expanduser('~/tmp/scale.npy'), scale)

    # load model
    print('Load parameters...')
    alpha = np.load(osp.expanduser('~/tmp/alpha.npy'))
    mu = np.load(osp.expanduser('~/tmp/mu.npy'))
    scale = np.load(osp.expanduser('~/tmp/scale.npy'))

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    gpu_options = tf.GPUOptions(allow_growth=True)
    cp = tf.ConfigProto(gpu_options=gpu_options)

    # sample
    print('Sampling...')
    t0 = time.time()
    # n = len(alpha)
    with tf.Session(config=cp) as sess:
        # draws = M.sample_mixtures(10, alpha[:n], mu[:n], scale[:n],
        #                           sample_func=sample_gaussian_numpy)
        # draws = M.sample_mixture_vectorized(10, alpha[:n], mu[:n], scale[:n],
        draws = M.sample_mixture_vectorized(100, alpha, mu, scale,
                                            sample_func=sample_gaussian_tril,
                                            debug=True)
    t1 = time.time()
    delta = t1 - t0
    print(f'Time used (seconds): {delta:,.2f}')
    print(draws.shape)
    print(draws)
