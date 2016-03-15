"""
Copyright (C) 2016 Baxter Eaves
License: Do what the fuck you want to public license (WTFPL) V2

Infinite Gaussian Mixture Model

Requires: igmm, numpy, scipy, matplotlib, seaborn
"""


import igmm
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from scipy.stats import chisquare

PTHRESH = .1


def pp_plot(f, p, nbins, ax=None):
    """ P-P plot of the empirical CDFs of values in two lists, f and p. """
    if ax is None:
        ax = plt.gca()

    uniqe_vals_f = list(set(f))
    uniqe_vals_p = list(set(p))

    combine = uniqe_vals_f
    combine.extend(uniqe_vals_p)
    combine = list(set(combine))

    if len(uniqe_vals_f) > nbins:
        bins = nbins
    else:
        bins = sorted(combine)
        bins.append(bins[-1]+bins[-1]-bins[-2])

    ff, edges = np.histogram(f, bins=bins, density=True)
    fp, _ = np.histogram(p, bins=edges, density=True)

    Ff = np.cumsum(ff*(edges[1:]-edges[:-1]))
    Fp = np.cumsum(fp*(edges[1:]-edges[:-1]))

    plt.plot([0, 1], [0, 1], c='dodgerblue', lw=2, alpha=.8)
    plt.plot(Ff, Fp, c='black', lw=2, alpha=.9)
    plt.xlim([0, 1])
    plt.ylim([0, 1])


class Geweke(object):
    def __init__(self, data_model, params, alpha, n_data, seed=None):
        self.model = data_model
        self.params = params
        self.alpha = alpha
        self.n = n_data

        self.stats_f = {'x_s': [], 'x_m': [], 'z_k': []}
        self.stats_p = {'x_s': [], 'x_m': [], 'z_k': []}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _insert_stats(self, forward):
        x_s = np.std(self.igmm.x)
        x_m = np.mean(self.igmm.x)
        z_k = self.igmm.k

        if forward:
            stats = self.stats_f
        else:
            stats = self.stats_p
        stats['x_s'].append(x_s)
        stats['x_m'].append(x_m)
        stats['z_k'].append(z_k)

    def forward_sample(self, no_collect=False):
        self.igmm = igmm.IGMM(np.random.rand(self.n), self.alpha, self.model,
                              self.params, gewke_forward=True)
        if not no_collect:
            self._insert_stats(True)

    def posterior_sample(self):
        self.igmm.infer(1)
        self.igmm.resample_data()
        self._insert_stats(False)

    def run(self, n_samples):
        for _ in range(n_samples):
            self.forward_sample()

        self.forward_sample(no_collect=True)
        for _ in range(n_samples):
            self.posterior_sample()

    @staticmethod
    def report_stat(p, stat_name):
        passtxt = 'FAILED'
        if p >= PTHRESH:
            passtxt = 'PASSED'
        print("%s: %s %s with p = %f" % (passtxt, stat_name, passtxt.lower(),
                                         p,))

    def report(self):
        ks_stats = ['x_s', 'x_m']
        x2_stats = ['z_k']

        n_stats = len(ks_stats) + len(x2_stats)

        plt.figure()

        c1, _, c2 = sns.color_palette('deep', 3)

        pt = 0
        for stat in ks_stats:
            pt += 1
            _, p = ks_2samp(self.stats_f[stat], self.stats_p[stat])
            Geweke.report_stat(p, stat)

            plt.subplot(2, n_stats, pt)
            sns.kdeplot(np.array(self.stats_f[stat]), color=c1, shade=True,
                        label='Forward')
            sns.kdeplot(np.array(self.stats_p[stat]), color=c2, shade=True,
                        label='Posterior')
            plt.xlabel(stat)

            plt.subplot(2, n_stats, n_stats+pt)
            pp_plot(self.stats_f[stat], self.stats_p[stat], 100)
            plt.xlabel('CDF %s forward' % stat)
            plt.ylabel('CDF %s posterior' % stat)

        for stat in x2_stats:
            pt += 1
            nbins = max(self.stats_f[stat] + self.stats_p[stat])+1
            bins_f = np.bincount(self.stats_f[stat], minlength=nbins)+1.
            bins_p = np.bincount(self.stats_p[stat], minlength=nbins)+1.
            _, p = chisquare(bins_f, bins_p)
            Geweke.report_stat(p, stat)

            plt.subplot(2, n_stats, pt)
            x = np.arange(nbins)
            plt.bar(x, bins_f, label='Forward', color=c1, alpha=.5)
            plt.bar(x, bins_p, label='Posterior', color=c2, alpha=.5)
            plt.xlabel(stat)

            plt.subplot(2, n_stats, n_stats+pt)
            pp_plot(self.stats_f[stat], self.stats_p[stat], nbins)
            plt.xlabel('CMF %s forward' % stat)
            plt.ylabel('CMF %s posterior' % stat)

        plt.suptitle('Geweke statistics')
        plt.show()


if __name__ == "__main__":
    # we need a high r and nu so that the std of the t distribution doesn't go
    # wild when the components are empty.
    params = (0., 121., 1., 98.,)
    alpha = 1.
    geweke = Geweke(igmm.NIG, params, alpha, 20)
    geweke.run(5000)
    geweke.report()
