# import torch
# import torch.nn as nn

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

        For multivariate MDN, shape is (U, V) where:
        U: number of data points
        V: number of mixtures
    mu : array
        Array of means for the mixtures.

        For multivariate MDN, shape is (U, V, D) where:
        U: number of data points
        V: number of mixtures
        D: dimension of each Gaussian used in the mixture.
    sigma : array
        Array of sigmas for the mixtures.

        For multivariate MDN, shape is (U, V, F) where:
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

    assert(V == alpha.shape[1])

    # weights should sum to 1.
    assert(np.allclose(alpha.sum(axis=1), 1.)), 'Alpha sums != 1.'

    # could generate a zero matrix. but since `sample_discrete()` needs
    # a random number, we generate here and use it later.
    # result is populated here.
    results = np.random.rand(U, n)
    samples = np.zeros((U, n, D))

    for i in range(U):
        # for each data sample
        for j in range(n):
            # draw n samples
            idx = sample_discrete(alpha[i], x=results[i, j])
            mu_draw = mu[i, idx]
            sigma_draw = sigma[i, idx]
            # form multivariate gaussian
            samples[i] = sample_func(mu_draw, sigma_draw, n)

    return samples


def sample_mixture_vectorized(n: int, alpha, mu, scale, sample_func,
                              debug=False):
    '''
    For each observation, make n samples of mu and scale pairs based on
    alpha as weights. Output shape (U, n).

    Construct vectorized mu_new and scale_new with those chosen indices.
    Reshape mu_new and scale_new to (-1, D) and (-1, F). Call length of mu_new
    as L = U * V.
    Sample from multivariate Gaussian.
    Results will have shape (n, L, D)

    Parameters
    ----------
    n : int

    alpha : TYPE

    mu : TYPE

    scale : TYPE

    sample_func : TYPE


    Returns
    -------
    TYPE
    '''
    U, V, D = mu.shape
    _, _, F = scale.shape

    assert(V == alpha.shape[1])

    # weights should sum to 1.
    assert(np.allclose(alpha.sum(axis=1), 1.)), 'Alpha sums != 1.'

    # could generate a zero matrix. but since `sample_discrete()` needs
    # a random number, we generate here and use it later.
    # result is populated here.

    # create alpha indices to iterate over, for each observation, n times.
    # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, ...z, z, z] for z + 1 observation
    # each 3 times.
    alpha_idx = [i for i in range(U) for _ in range(n)]

    results = np.random.rand(U * n)
    expected_samples = len(results)

    # generate mixture index list from discrete weighted sampling
    # for each data sample, we draw n index positions each indicating which
    # Gaussian in the mixture is used.
    idx = [sample_discrete(alpha[alpha_idx[z]], x=results[z])
           for z in range(len(results))]
    assert(len(idx) == expected_samples), \
        f'Expecting {expected_samples}, got {len(idx)}.'

    mu_new = mu.reshape(-1, D)[idx, :]
    scale_new = scale.reshape(-1, F)[idx, :]

    if debug:
        print('Expected Samples: ', expected_samples)
        print('mu_new.shape: ', mu_new.shape)
        print('scale_new.shape:', scale_new.shape)

    assert(mu_new.shape[0] == scale_new.shape[0] == expected_samples)

    # from each distribution, draw 1 sample.
    samples = sample_func(mu_new, scale_new, n=1)

    if debug:
        print('samples.shape: ', samples.shape)

    assert(samples.shape[1] == expected_samples), \
        f'Expecting {expected_samples} but got {len(samples)}.'
    assert(samples.shape[-1] == D), \
        f'Expecting sample dimension {D}, got{samples.shape[-1]}'

    # reshape back to match input dimension format, i.e. for each observation,
    # n samples of dimension D.
    samples = samples.reshape(U, n, D)

    return samples
