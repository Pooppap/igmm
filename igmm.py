"""
Copyright (C) 2016 Baxter Eaves
License: Do what the fuck you want to public license (WTFPL) V2

Infinite Gaussian Mixture Model

Requires: numpy, scipy, matplotlib, seaborn, pandas
"""

import numpy as np

from random import shuffle
from scipy.special import gammaln
from scipy.misc import logsumexp
from math import log
from math import pi

LOG2 = log(2)
LOG2PI = log(2*pi)
LOGSQRTPI = .5*log(pi)
LOGSQRT2PI = .5*log(2*pi)


class ConjugateMixtureComponent(object):
    def __init__(self, x, *params):
        raise NotImplementedError

    def posterior_predictive(self, x):
        raise NotImplementedError

    def marginal_likelihood(self):
        raise NotImplementedError

    def insert_datum(x):
        raise NotImplementedError

    def remove_datum(x):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError


class NIG(ConjugateMixtureComponent):
    """ Normal inverse-Gamma component model """
    def __init__(self, x, m, r, s, nu):
        self.m = m
        self.r = r
        self.s = s
        self.nu = nu

        self.logz = self._psi(r, s, nu)

        self.n = 0.
        self.sum_x = 0.
        self.sum_x_sq = 0.

        # TODO: put this in superclass constructor?
        if x is not None:
            for x_i in x:
                self.insert_datum(x_i)

    def posterior_predictive(self, x):
        (mn, rn, sn, nun) = self._update_params(
            self.n, self.sum_x, self.sum_x_sq, self.m, self.r, self.s, self.nu)
        (_, rm, sm, num) = self._update_params(1, x, x*x, mn, rn, sn, nun)
        return -LOGSQRT2PI + self._psi(rm, sm, num) - self._psi(rn, sn, nun)

    def marginal_likelihood(self):
        (mn, rn, sn, nun) = self._update_params(
            self.n, self.sum_x, self.sum_x_sq, self.m, self.r, self.s, self.nu)
        return -(self.n/2.)*LOG2PI + self._psi(rn, sn, nun) - self.logz

    def insert_datum(self, x):
        self.n += 1
        self.sum_x += x
        self.sum_x_sq += x*x

    def remove_datum(self, x):
        self.n -= 1
        self.sum_x -= x
        self.sum_x_sq -= x*x

    def draw(self):
        if self.n > 0:
            (m, r, s, nu,) = self._update_params(
                self.n, self.sum_x, self.sum_x_sq, self.m, self.r, self.s,
                self.nu)
        else:
            (m, r, s, nu,) = (self.m, self.r, self.s, self.nu,)

        coeff = ((.5*s*(r+1.0)) / (.5*nu*r))**.5
        draw = np.random.standard_t(nu)*coeff + m

        return draw

    def __repr__(self):
        print("NIG (Normal, inverse-gamma)")
        print(" Parameters:")
        print(" - m: %f" % self.m)
        print(" - r: %f" % self.r)
        print(" - s: %f" % self.s)
        print(" - nu: %f" % self.nu)
        print(" Sufficient statistics:")
        print(" - n: %d" % self.n)
        print(" - sum_x: %f" % self.sum_x)
        print(" - sum_x_sq: %f" % self.sum_x_sq)

    @property
    def params(self):
        return self.m, self.r, self.s, self.nu

    @staticmethod
    def _psi(r, s, nu):
        logz = (nu+1.)/2.*LOG2 + LOGSQRTPI - .5*log(r) - nu/2.*log(s)
        logz += gammaln(nu/2.)
        return logz

    @staticmethod
    def _update_params(n, sum_x, sum_x_sq, m, r, s, nu):
        rn = r + n
        nun = nu + n
        mn = (r*m + sum_x)/rn
        sn = s + sum_x_sq + r*m*m - rn*mn*mn

        assert sn > 0

        return mn, rn, sn, nun


class IGMM(object):
    """ Infinite Gaussian Mixture Model """
    def __init__(self, x, alpha, component_model, params, seqinit=False,
                 gewke_forward=False):
        self.x = x
        self.n = len(x)
        self.alpha = alpha
        self.params = params
        self.model = component_model

        if seqinit:
            self.z = np.zeros(self.n, dtype=int)
            self.nk = [1]
            self.k = 1
            self.components = [self.model([self.x[0]], *params)]
            for i in range(1, self.n):
                self.step(i, init_mode=True, )
        else:
            self.z, self.nk, self.k = crp_gen(self.n, alpha)
            self.components = []
            if gewke_forward:

                # init empty components
                x_tmp = []
                for j in range(self.k):
                    self.components.append(self.model(None, *params))

                # insert new data
                for j in self.z:
                    x_i = self.components[j].draw()
                    self.components[j].insert_datum(x_i)
                    x_tmp.append(x_i)

                self.x = np.array(x_tmp)

            else:
                for j in range(self.k):
                    x_j = self.x[self.z == j]
                    self.components.append(self.model(x_j, *params))

    def regen(self):
        self.z = np.zeros(self.n, dtype=int)
        self.nk = [1]
        self.k = 1
        self.components = [self.model([self.x[0]], *self.params)]
        q = 0.
        for i in range(1, self.n):
            q += self.step(i, init_mode=True, return_prob=True)
        return q, self.z

    def _remove_datum(self, idx):
        """ Remove a datum from a component """
        assert idx < self.n

        j = self.z[idx]
        is_singleton = self.nk[j] == 1
        self.z[idx] = -1
        if is_singleton:
            self.k -= 1
            del self.nk[j]
            self.z[self.z > j] -= 1
            del self.components[j]
        else:
            self.components[j].remove_datum(self.x[idx])
            self.nk[j] -= 1

        assert len(self.nk) == self.k
        assert len(self.components) == self.k
        # assert sum(self.nk) == self.n-1
        assert max(self.z) == self.k-1
        assert min(self.z) == -1
        assert sum(self.z < 0) == 1

    def _insert_datum(self, idx, j):
        """ Insert a datum into a component """
        assert idx < self.n
        assert j <= self.k

        self.z[idx] = j
        if j == self.k:
            self.nk.append(1)
            self.k += 1
            self.components.append(self.model([self.x[idx]], *self.params))
        else:
            self.nk[j] += 1
            self.components[j].insert_datum(self.x[idx])

        assert len(self.nk) == self.k
        assert len(self.components) == self.k
        # assert sum(self.nk) == self.n
        assert max(self.z) == self.k-1
        assert min(self.z) == 0
        assert sum(self.z < 0) == 0

    def step(self, idx, init_mode=False, return_prob=False):
        """ Resample the component assignment of x_idx """
        assert idx < self.n

        if not init_mode:
            self._remove_datum(idx)

        logps = np.log(np.array(self.nk + [self.alpha]))

        x_i = self.x[idx]
        for j, component in enumerate(self.components):
            logps[j] += component.posterior_predictive(x_i)

        temp_component = self.model([x_i], *self.params)
        logps[-1] += temp_component.marginal_likelihood()

        z_i = logpflip(logps)
        self._insert_datum(idx, z_i)

        if return_prob:
            logps -= logsumexp(logps)
            return logps[z_i]

    def infer(self, n_sweeps):
        """ Run the Gibbs sampler for n_sweeps. Each sweep resamples each
        datum.
        """
        idxs = [i for i in range(self.n)]
        for i in range(n_sweeps):
            shuffle(idxs)
            for idx in idxs:
                self.step(idx)

    def resample_data(self):
        """ Resample data conditional on existing data. """
        order = [i for i in range(self.n)]
        shuffle(order)
        for i in order:
            j = self.z[i]
            x_i = self.x[i]
            self.components[j].remove_datum(x_i)
            x_i = self.components[j].draw()
            self.components[j].insert_datum(x_i)
            self.x[i] = x_i


def pflip(w):
    """ Normalize the weights, w, and do a categorical draw """
    p = np.cumsum(w)
    p /= p[-1]
    return np.digitize([np.random.rand()], p)[0]


def logpflip(lw):
    """ Normalize log weights, lw, and do a categorical draw """
    p = np.cumsum(np.exp(lw-logsumexp(lw)))
    return np.digitize([np.random.rand()], p)[0]


def crp_gen(n, alpha):
    """ Draw an n-length partition from CRP(alpha) """
    if alpha <= 0.0:
        raise ValueError('alpha must be greater than 0.')

    z = np.zeros(n, dtype=int)
    nk = [1]
    k = 1
    for i in range(1, n):
        j = pflip(np.array(nk + [alpha]))

        assert j <= k

        z[i] = j
        if j == k:
            k += 1
            nk.append(1)
        else:
            nk[j] += 1

    shuffle(z)

    assert len(z) == n
    assert len(nk) == k
    assert sum(nk) == n
    assert min(z) == 0
    assert max(z) == k-1

    return z, nk, k


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    sns.set_context('paper')

    n = 50
    std = .75
    x = np.append(np.random.randn(n)*std,
                  (np.random.randn(n)*std+4., np.random.randn(n)*std-4.,))

    igmm = IGMM(x, .5, NIG, (0., 1., 1., 1.,), seqinit=True)

    plt.figure(figsize=(7.5, 3.5), dpi=150)

    igmm.infer(100)
    z = igmm.z
    k = igmm.k

    df = pd.DataFrame([{'x': xi, 'j': j} for j, xi in zip(z, x)])

    plt.clf()
    plt.subplot(1, 2, 1)
    sns.distplot(df['x'], bins=35, hist=True, rug=True, kde=False)
    plt.title('Raw Data')

    plt.subplot(1, 2, 2)
    for j in range(k):
        xj = df['x'][df['j'] == j]
        if len(xj) == 1:
            continue
        sns.distplot(xj, hist=False, rug=True, kde=True, norm_hist=False)
    plt.title('Categorized Data')
    plt.show()
