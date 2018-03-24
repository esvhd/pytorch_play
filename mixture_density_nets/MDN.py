import torch
import torch.nn as nn

import numpy as np


def sample_discrete(pdf, x=None):
    '''
    Alias method to draw samples from discrete distributions based on
    uneven probabilities.

    This method is slightly faster than the original one. 200ms vs 360ms,
    with 20 discrete categories.

    Parameters:
    ------------------------------
    pdf: numpy.ndarray
        Probablity density, must be all positive and sum to 1. This defines
        the probabilities of the discrete distribution. In repeated draws,
        the returned index result will approximate this probability
        distribution.
    x: float
        probability, must be non-negative. If not given a random number
        is generated.

    Returns:
        One sample of index position.
    '''
    assert(np.all(pdf >= 0)), 'Not all elements are >= 0.'

    cs = pdf.cumsum()
    assert(np.isclose(cs[-1], 1)), 'PDF does not sum to 1.'

    if x is None:
        # generate uniform random number [0, 1)
        x = np.random.rand()
    else:
        assert x >= 0

    idx = np.where(cs >= x)[0]
    if len(idx) < 1:
        print('error with sampling ensemble')
        return -1
    else:
        return idx[0]


def sample_mixtures(n: int, alpha, mu, sigma, sample_func):
    '''
    Sample from a mixture density network.

    Parameters
    ----------
    n : int
        number of samples to draw at each point.
    alpha : array
        Alpha parameter from MDN, i.e. weights for the mixtures. Samples are
        drawn based on the weights defined in this array.
    mu : array
        Array of means for the mixtures.

        For multivariate MDN, shape is
        (U, V, D) where:
        U: number of data points
        V: number of mixtures
        D: dimension of each Gaussian used in the mixture.
    sigma : array
        Array of sigmas for the mixtures.

        For multivariate MDN, shape is:
        (U, V, F) where:
        U, V are same as those specified in mu.
        F: lower trianglar covariance matrix in 1D array format.
    sample_func : TYPE
        Gaussian sample function. Left undefined so that it can suit different
        ML frameworks. E.g. pytorch or tensorflow.

    Returns
    -------
    TYPE

    '''
    U, V, D = mu.shape
    _, _, F = sigma.shape

    assert(V == len(alpha))

    # weights should sum to 1.
    assert(np.isclose(np.sum(alpha), 1.))

    # could generate a zero matrix. but since `sample_discrete()` needs
    # a random number, we generate here and use it later.
    # result is populated here.
    results = np.random.rand(U, n)

    for i in range(U):
        # for each data sample
        for j in range(n):
            # draw n samples
            idx = sample_discrete(alpha, x=results[i, j])
            mu_draw = mu[i, idx]
            sigma_draw = sigma[i, idx]
            # form multivariate gaussian
            results[i, j] = sample_func(mu_draw, sigma_draw)

    return results
