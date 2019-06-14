#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import matplotlib.pyplot as plt
import iminuit as im


class Evol(lssfr_med, stretch_name, stretch_err_name, lssfr_name):
    ''' '''

    FREEPARAMETERS = ['mu_y', 'sigma_y', 'mu_o', 'sigma_o']

    ################################### SETTER ################################

    def set_data(self, pandas):
        '''Donne les pandas des stretch young & old'''
        self.pandas_y = pandas.loc[pandas['lssfr_name'] > lssfr_med]
        self.pandas_o = pandas.loc[pandas['lssfr_name'] < lssfr_med]
        self.stretch_y_err = self.pandas_y['stretch_err_name']
        self.stretch_o_err = self.pandas_o['stretch_err_name']
        self.stretch_y = self.pandas_y['stretch_name']
        self.stretch_o = self.pandas_o['stretch_name']
        self.lssfr_y = self.pandas_y['lssfr_name']
        self.lssfr_o = self.pandas_o['lssfr_name']

    def set_param(self, param):
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}


    ################################### FITTER ################################

    def distro(self, x, dx, mu, sigma):
        '''Le modèle de distribution'''
        return scipy.stats.norm.pdf(x, mu, scale=np.sqrt(dx**2+sigma**2))

    def min_distro_y(self, mu, sigma):
        '''La fonction à minimiser pour young'''
        return -2*np.sum(np.log(self.distro(self.stretch_y, self.stretch_y_err,
                                            mu, sigma)))

    def min_distro_o(self, mu, sigma):
        '''La fonction à minimiser pour old'''
        return -2*np.sum(np.log(self.distro(self.stretch_o, self.stretch_o_err,
                                            mu, sigma)))

    def minimize(self, param_guess):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m = im.Minuit(self.get_logproba, param = param_guess,
                           print_level=0, pedantic=False,
                           use_array_call=True,
                           forced_parameters="param")
                           #limit_param=[0,None])
        self.m.migrad();
        self.set_param([self.m.values['a'], self.m.values['b'], \
                        self.m.values['zmax'], self.m.values['zc']], True)


    def minimize(self):
        '''Actual minimization'''
        m_y = im.Minuit(self.min_distro_y, mu=0, sigma=1,
                        print_level=0, pedantic=False)
        m_o = im.Minuit(self.min_distro_o, mu=0, sigma=1,
                        print_level=0, pedantic=False)

        m_y.migrad()
        m_o.migrad()

        self.params['mu_y'] = m_y.values['mu']

    ################################### PLOTTER ##################################

    def plt_data(self,c,l):
        '''Trace l\'histogramme des data binées'''
        plt.figure(figsize = (17,6))
        plt.hist(self.rawdata, bins = self.bord, color = c, label = l, histtype = 'step')

        ax = plt.gca()
        ax.tick_params(axis = 'both',
                       direction = 'in',
                       length = 10, width = 3,
                       labelsize = 20,
                       which = 'both',
                       top = True, right = True)
        plt.xlabel('z', fontsize = 20)
        plt.ylabel('# of SNeIa', fontsize = 20)
        plt.title('Data for ' + str(len(self.bins)) + ' bins and zmax = ' + str(self.rawdata[-1]), fontsize = 20)
        plt.legend()

    def plt_data_fit(self,c,l):
        '''Trace l\'histogramme des data binées, le fit et les nombres de SNe'''
        plt.figure(figsize = (17,6))
        plt.hist(self.rawdata, bins = self.bord, color = c, label = l, histtype = 'step')

        ax = plt.gca()
        lim = ax.axis()
        ax.tick_params(axis = 'both',
                      direction = 'in',
                      length = 10, width = 3,
                      labelsize = 20,
                      which = 'both',
                      top = True, right = True)
        lnspc = np.linspace(lim[0], lim[1], len(self.rawdata))
        ax.plot(lnspc, self.get_ratemodel(lnspc, usebestfit = True), label='fit', color = c)

        centers = [self.bins[i][0] + np.diff(self.bins[i])/2 for i in range(len(self.bins))]

        ax.scatter(centers, self.data,
                   s = 80, c = c, edgecolors = 'face',
                   alpha = .8, label = 'data')

        ax.errorbar(centers, self.data,
                    yerr = np.sqrt(self.data),
                    marker = 'None', ls = 'None',
                    ecolor = "0.7")

        ax.scatter(centers, self.get_nbSN(), marker = '+', s = 80)

        ax.axis(lim)

        plt.xlabel('z', fontsize = 20)
        plt.ylabel('# of SNeIa', fontsize = 20)
        plt.title('Fit for ' + str(len(self.bins)) + ' bins and zmax = ' + str(self.rawdata[-1]), fontsize = 20)

        plt.legend()
