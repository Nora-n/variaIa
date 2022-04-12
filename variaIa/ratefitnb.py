#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
import iminuit as im

dict_zmax = {'SDSS': [0.1728, 0.2039, 0.2301],
             'PS1': [0.1821, 0.2732, 0.3568],
             'SNLS': [0.5937, 0.6162, 0.6565]}

# =========================================================================== #
#                                                                             #
#                                   RATEFIT                                   #
#                                                                             #
# =========================================================================== #


class RateFit():
    ''' '''

    # =================================================================== #
    #                               Parameters                            #
    # =================================================================== #

    FREEPARAMETERS = ['a', 'zmax']
    VOLUME_SCALE = 1e8

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

    def set_data(self, dataz, survey):
        '''Donne la liste du nombre observé de SNe par bin'''
        self.data = np.sort(dataz)
        self.counts = np.asarray([(i+1)*1 for i in range(len(dataz))]).T
        self.survey = survey

    def set_param(self, param):
        '''Créé le dico des noms des params et de leurs valeurs'''
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def get_cuts(self, zmax):
        '''Implémentation de zmax pour le fit'''
        zwhere = np.where(self.data < zmax)
        res = {'dat': self.data[zwhere],
               'counts': self.counts[zwhere]}
        return(res)

    def get_ratemodel(self, z, a):
        ''' Le modèle en question'''
        return a/self.VOLUME_SCALE * cosmo.comoving_volume(z).value

    def get_probabilities(self, a, zmax):
        """ Returns the poisson statistics applied to each cases """
        return stats.poisson.pmf(self.get_cuts(zmax)['counts'],
                                 self.get_ratemodel(self.get_cuts(zmax)['dat'],
                                                    a))

    def get_loglikelihood(self, a, zmax):
        """ """
        # poisson.pmf(k, mu) = probability to observe k counts\
        # given that we expect mu
        return -np.sum(np.log(self.get_probabilities(a, zmax)))

    def get_loglikeDOF(self, a, zmax):
        """Test of better minimization function"""
        k = len(self.FREEPARAMETERS)
        n = len(self.get_cuts(zmax)['dat'])
        DoF = n-k
        return(self.get_loglikelihood(a, zmax)/DoF)

    def get_aicc(self, a, zmax):
        k = len(self.FREEPARAMETERS)
        n = len(self.get_cuts(zmax)['dat'])
        logl = self.get_loglikelihood(a, zmax)

        return(2*k + logl + (2*k*(k+1))/(n-k-1))

    def get_aiccDOF(self, a, zmax):
        """To have high zmax even with lots of data"""
        k = len(self.FREEPARAMETERS)
        n = len(self.get_cuts(zmax)['dat'])
        DoF = n-k
        return(self.get_aicc(a, zmax)/DoF)

    def get_logprior(self, a, zmax):
        '''Donne la probabilité du modèle (limites sur "a", "zmax")'''
        if np.logical_and(a > 1e-1,
                          a < 6e1):
            prior_bounds_a = 0
        else:
            prior_bounds_a = 1e10

        if np.logical_and(zmax >= 1e-1,
                          zmax < 7e-1):
            prior_bounds_zmax = 0
        else:
            prior_bounds_zmax = 1e10

        prior_zmax = -np.log(stats.norm.pdf(zmax,
                                            dict_zmax[self.survey][1],
                                            .05))

#        if abs(self.get_cuts(zmax)['counts'][-1]
#                - self.get_ratemodel(self.get_cuts(zmax)['dat'][-1], a)) < 7:
#            prior_counts = 0
#        else:
#            prior_counts = 1e10

        used_priors = [prior_bounds_a, prior_bounds_zmax, prior_zmax]
        return np.sum(used_priors)

    def get_logprob(self, a, zmax):
        """ """
        return self.get_loglikeDOF(a, zmax) + self.get_logprior(a, zmax)

    def minimize(self, limits, guess_a, guess_zmax):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m = im.Minuit(self.get_logprob, a=guess_a, zmax=guess_zmax)
        self.m.limits = limits

        self.migrad_out = self.m.migrad()

        self.set_param([round(self.m.values['a'], 4),
                        round(self.m.values['zmax'], 4)])

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

    def plot(self, guess=None):
        '''Trace les données vs le modèle, et modèle fitté (optionnel)'''

        colors = {'SDSS': 'lime',
                  'PS1': 'blue',
                  'SNLS': 'red'}

        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        zz = np.linspace(0, np.max(self.data))

        ax.plot(self.data,
                self.counts,
                color=colors[self.survey],
                lw=1.0, label=self.survey)

        ax.plot(zz,
                self.get_ratemodel(zz, self.param['a']),
                color='black',
                lw=1.0, label='expected')

        ax.vline(self.param['zmax'],
                 color=".3",
                 lw=1.0, label="zmax")
        if guess is not None:
            ax.plot(zz,
                    self.get_ratemodel(zz, guess[0]),
                    color='orange',
                    lw=1.0, label='expected_guess')

            ax.vline(guess[-1],
                     color="orange",
                     lw=1.0, label="zmax_guess")

        zerr = self.migrad_out.errors[-1]

        ax.vspan(self.param['zmax']-zerr,
                 self.param['zmax']+zerr,
                 color=".3", alpha=.2,
                 lw=1.0)

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=12,
                       top=True, right=True)

        ax.set_xlim(0, np.max(self.data))
        ax.set_ylim(0, 500)

        ax.set_xlabel(r'$\mathrm{redshift}$', fontsize='x-large')
        ax.set_ylabel(r'$\mathrm{N}_\mathrm{SNe\,Ia}$', fontsize='x-large')

        plt.legend(ncol=1, loc='lower right')

        plt.title(r'$\mathrm{Evolution\,\,of\,\,poisson\,\,cdf\,\,with\,\,}$' +
                  r'$\mathrm{median\,\,for\,\,}$' + self.survey,
                  fontsize='x-large')
