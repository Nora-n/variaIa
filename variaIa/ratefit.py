#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import poisson
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
import iminuit as im


class RateFit():
    ''' '''

    FREEPARAMETERS = ['a', 'b', 'zmax', 'zc']
    VOLUME_SCALE = 1e8

# =========================================================================== #
#                                    SETTER                                   #
# =========================================================================== #

    def set_data(self, datap, bins):
        '''Donne la liste du nombre observé de SNe par bin'''
        self.rawdata = np.sort(datap)
        self.bins = np.asarray(bins)
        self.bord = np.append(bins[:, 0], bins[-1, 1])
        self.data, _ = np.histogram(self.rawdata, self.bord)

    def set_param(self, param, isbestfit=False):
        '''Créé le dico des noms des params et de leurs valeurs'''
        if isbestfit:
            self.bestparam = {k: v for k, v in zip(self.FREEPARAMETERS,
                                                   np.atleast_1d(param))}
        else:
            self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                               np.atleast_1d(param))}

# =========================================================================== #
#                                    FITTER                                   #
# =========================================================================== #

    def get_ratemodel(self, z, usebestfit=False):
        ''' Le modèle en question'''
        if usebestfit:
            return self.bestparam['a'] * \
                cosmo.comoving_volume(z).value/self.VOLUME_SCALE
        else:
            return self.param['a'] * \
                cosmo.comoving_volume(z).value/self.VOLUME_SCALE

    def get_missedSN(self, z, usebestfit=False):
        '''Le modèle de décroissance de SNe'''
        return np.append(np.zeros(len(z[np.where(z < self.param['zmax'])])),
                         self.param['b'] *
                         np.exp(z[np.where(z > self.param['zmax'])] /
                                self.param['zc'])
                         / self.VOLUME_SCALE)

    def get_nbSN(self, bins=None):
        '''Donne le nombre de SNe entre deux redshifts
        calculé à partir du modèle'''
        if bins is None:
            bins = self.bins
        return self.get_ratemodel(self.bins[:, 1]) -\
            self.get_ratemodel(self.bins[:, 0]) +\
            self.get_missedSN(self.bins[:, 1]) -\
            self.get_missedSN(self.bins[:, 0])

    def get_loglikelihood(self, param=None):
        '''Donne le likelihood sur la totalité des données'''
        if param is not None:
            self.set_param(param)
        return -np.sum(np.log(poisson.pmf(self.data, self.get_nbSN())))

    def get_logprior(self, param=None):
        '''Donne la probabilité du modèle (limites sur "a")'''
        if param is not None:
            self.set_param(param)
        prior_bounds = 0 if np.logical_and(self.param["a"] > 1e-4,
                                           self.param["a"] < 1e2) else 1e10
        prior_somethingelse = 0
        used_priors = [prior_bounds, prior_somethingelse]
        return np.sum(used_priors)

    def get_logproba(self, param=None):
        """ """
        if param is not None:
            self.set_param(param)
        return self.get_loglikelihood() + self.get_logprior()

    def minimize(self, param_guess):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m = im.Minuit(self.get_logproba, param=param_guess)
        # limit_param=[0,None])
        self.m.migrad()
        self.set_param([self.m.values['a'], self.m.values['b'],
                        self.m.values['zmax'], self.m.values['zc']], True)

# =========================================================================== #
#                                    PLOTER                                   #
# =========================================================================== #

    def plt_data(self, c, label):
        '''Trace l\'histogramme des data binées'''
        plt.figure(figsize=(17, 6))
        plt.hist(self.rawdata, bins=self.bord,
                 color=c, label=label, histtype='step')

        ax = plt.gca()
        ax.tick_params(axis='both',
                       direction='in',
                       length=10, width=3,
                       labelsize=20,
                       which='both',
                       top=True, right=True)
        plt.xlabel('z', fontsize=20)
        plt.ylabel('# of SNeIa', fontsize=20)
        plt.title('Data for ' + str(len(self.bins)) +
                  ' bins and zmax = ' + str(self.rawdata[-1]), fontsize=20)
        plt.legend()

    def plt_data_fit(self, c, label):
        """Trace l'histogramme des data binées, le fit et les nombres de SNe"""
        plt.figure(figsize=(17, 6))
        plt.hist(self.rawdata, bins=self.bord,
                 color=c, label=label, histtype='step')

        ax = plt.gca()
        lim = ax.axis()
        ax.tick_params(axis='both',
                       direction='in',
                       length=10, width=3,
                       labelsize=20,
                       which='both',
                       top=True, right=True)
        lnspc = np.linspace(lim[0], lim[1], len(self.rawdata))
        ax.plot(lnspc, self.get_ratemodel(
            lnspc, usebestfit=True), label='fit', color=c)

        centers = [self.bins[i][0] +
                   np.diff(self.bins[i])/2 for i in range(len(self.bins))]

        ax.scatter(centers, self.data,
                   s=80, c=c, edgecolors='face',
                   alpha=.8, label='data')

        ax.errorbar(centers, self.data,
                    yerr=np.sqrt(self.data),
                    marker='None', ls='None',
                    ecolor="0.7")

        ax.scatter(centers, self.get_nbSN(), marker='+', s=80)

        ax.axis(lim)

        plt.xlabel('z', fontsize=20)
        plt.ylabel('# of SNeIa', fontsize=20)
        plt.title('Fit for ' + str(len(self.bins)) +
                  ' bins and zmax = ' + str(self.rawdata[-1]), fontsize=20)

        plt.legend()
