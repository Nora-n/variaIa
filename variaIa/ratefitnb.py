#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
import iminuit as im


class RateFit():
    ''' '''

    FREEPARAMETERS = ['a', 'zmax']
    VOLUME_SCALE = 1e8

#   ################################### SETTER ################################

    def set_data(self, dataz):
        '''Donne la liste du nombre observé de SNe par bin'''
        self.data = np.sort(dataz)
        self.counts = np.asarray([(i+1)*1 for i in range(len(dataz))]).T

    def set_param(self, param):
        '''Créé le dico des noms des params et de leurs valeurs'''
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

#   ################################### FITTER ################################

    def get_cuts(self, zmax):
        '''Implémentation de zmax pour le fit'''
        zwhere = np.where(self.data < zmax)
        return(self.data[zwhere], self.counts[zwhere])

    def get_ratemodel(self, z, a):
        ''' Le modèle en question'''
        return a/self.VOLUME_SCALE * cosmo.comoving_volume(z).value

    def get_loglikelihood(self, a, zmax):
        '''Donne le likelihood sur la totalité des données'''
        return -np.sum((self.get_cuts(zmax)[1]
                       - self.get_ratemodel(self.get_cuts(zmax)[0], a))**2)

    def get_aicc(self, a, zmax):
        k = len(self.FREEPARAMETERS)
        logl = self.get_loglikelihood(a, zmax)

        return 2*k + logl + (2*k*(k+1))/(len(self.get_cuts(zmax)[0])-k-1)

    def get_logprior(self, a, zmax):
        '''Donne la probabilité du modèle (limites sur "a", "zmax")'''
        if np.logical_and(a > 1e-1,
                          a < 1e1):
            prior_bounds_a = 0
        else:
            prior_bounds_a = 1e10

        if np.logical_and(zmax > 1e-1,
                          zmax < 7e-1):
            prior_bounds_zmax = 0
        else:
            prior_bounds_zmax = 1e10

        used_priors = [prior_bounds_a, prior_bounds_zmax]
        return np.sum(used_priors)

    def get_logproba(self, a, zmax):
        """ """
        return self.get_aicc(a, zmax) + self.get_logprior(a, zmax)

    def minimize(self, **kwargs):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m = im.Minuit(self.get_logproba, **kwargs,
                           print_level=0, pedantic=False)

        self.migrad_out = self.m.migrad()

        self.set_param([self.m.values['a'],
                        self.m.values['zmax']])

#   ################################### PLOTTER ###############################

    def plot(self, label):
        '''Trace les données vs le modèle, et modèle fitté (optionnel)'''
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        ax.plot(self.data,
                self.counts,
                color='blue',
                lw=1.0, label=label)

        ax.plot(self.data,
                self.get_ratemodel(self.data, self.param['a']),
                color='black',
                lw=1.0, label='expected')

        ax.vline(self.param['zmax'],
                 color=".3",
                 lw=1.0, label="zmax")

        zerr = self.migrad_out[1][1][3]

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

        plt.legend(ncol=1, loc='upper right')

        plt.title(r'$\mathrm{Evolution\,\,of\,\,poisson\,\,cdf\,\,with\,\,}$' +
                  r'$\mathrm{median\,\,for\,\,surveys}$',
                  fontsize='x-large')
