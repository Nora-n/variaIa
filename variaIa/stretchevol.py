#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import iminuit as im
import matplotlib.pyplot as plt
plt.style.use(['classic', 'seaborn-white'])

lssfr_med = -10.8

'''USAGE :
names = {'lssfr_name':       'lssfr',
         'stretch_name':     'salt2.X1',
         'stretch_err_name': 'salt2.X1.err',
         'lssfr_err_d_name': 'lssfr.err_down',
         'lssfr_err_u_name': 'lssfr.err_up',
         'py_name':          'p(prompt)}

evolD = stretchevol.EvolDouble()
evolD.set_names(names)
evolD.set_data(pandas)

evolD.minimize()

evolD.plt_scatter()
'''

#   ###########################################################################
#   ################################ EVOLSIMPLE ###############################
#   ###########################################################################


class EvolSimple():
    ''' '''

    FREEPARAMETERS = ['mu_y', 'sigma_y', 'mu_o', 'sigma_o']

#   ################################# SETTER ##################################

    def set_names(self, names):
        '''Permet les extractions sur les pandas
        names = {'lssfr_name':       'lssfr',
                 'stretch_name':     'salt2.X1',
                 'stretch_err_name': 'salt2.X1.err',
                 'lssfr_err_d_name': 'lssfr.err_down',
                 'lssfr_err_u_name': 'lssfr.err_up'}'''
        self.names = names

    def set_data(self, pandas):
        '''Donne les pandas des stretch young & old'''
        self.pandas_y = pandas.loc[pandas[self.names['lssfr_name']] > lssfr_med]
        self.pandas_o = pandas.loc[pandas[self.names['lssfr_name']] < lssfr_med]
        self.stretch_y_err = self.pandas_y[self.names['stretch_err_name']]
        self.stretch_o_err = self.pandas_o[self.names['stretch_err_name']]
        self.stretch_y = self.pandas_y[self.names['stretch_name']]
        self.stretch_o = self.pandas_o[self.names['stretch_name']]
        self.lssfr_y = self.pandas_y[self.names['lssfr_name']]
        self.lssfr_o = self.pandas_o[self.names['lssfr_name']]

        self.stretch = pandas[self.names['stretch_name']]
        self.stretch_err = pandas[self.names['stretch_err_name']]
        self.lssfr = pandas[self.names['lssfr_name']]
        self.lssfr_err_d = pandas[self.names['lssfr_err_d_name']]
        self.lssfr_err_u = pandas[self.names['lssfr_err_u_name']]

        self.floor = np.floor(np.min(self.stretch)-0.4)
        self.ceil = np.ceil(np.max(self.stretch)+0.4)

    def set_param(self, param):
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

#   ################################## FITTER #################################

    def gauss(self, x, dx, mu, sigma):
        '''Le modèle de distribution'''
        return scipy.stats.norm.pdf(x, mu, scale=np.sqrt(dx**2+sigma**2))

    def min_gauss_y(self, mu_y, sigma_y):
        '''La fonction à minimiser pour young'''
        return -2*np.sum(np.log(self.gauss(self.stretch_y, self.stretch_y_err,
                                           mu_y, sigma_y)))

    def min_gauss_o(self, mu_o, sigma_o):
        '''La fonction à minimiser pour old'''
        return -2*np.sum(np.log(self.gauss(self.stretch_o, self.stretch_o_err,
                                           mu_o, sigma_o)))

    def minimize(self, mu_y_guess=False, sigma_y_guess=False,
                 mu_o_guess=False, sigma_o_guess=False):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m_y = im.Minuit(self.min_gauss_y, mu_y=mu_y_guess,
                             sigma_y=sigma_y_guess,
                             print_level=0, pedantic=False)

        self.m_o = im.Minuit(self.min_gauss_o, mu_o=mu_o_guess,
                             sigma_o=sigma_o_guess,
                             print_level=0, pedantic=False)

        self.m_y.migrad()
        self.m_o.migrad()

        self.set_param([self.m_y.values['mu_y'], self.m_y.values['sigma_y'],
                        self.m_o.values['mu_o'], self.m_o.values['sigma_o']])

#   ################################# PLOTTER #################################

    def plt_scatter(self):
        '''Trace le nuage de points et les fits'''
        plt.scatter(self.lssfr_y,
                    self.stretch_y,
                    marker='o', s=20,
                    color='purple', label='young PG')
        plt.scatter(self.lssfr_o,
                    self.stretch_o,
                    marker='o', s=20,
                    color='orange', label='old PG')

        plt.errorbar(self.lssfr,
                     self.stretch,
                     xerr=[self.lssfr_err_d, self.lssfr_err_u],
                     yerr=self.stretch_err,
                     ecolor='gray', alpha=.3,
                     ls='none', label=None)

        plt.plot([lssfr_med, lssfr_med],
                 [self.floor, self.ceil],
                 color='b', alpha=.5, linewidth=2.0)

        x_linspace = np.linspace(self.floor, self.ceil, 1000)

        plt.plot(self.gauss(x_linspace, 0, self.param['mu_y'],
                            self.param['sigma_y'])+lssfr_med,
                 x_linspace,
                 color='r', label='gaussian young')
        plt.plot(-self.gauss(x_linspace, 0, self.param['mu_o'],
                             self.param['sigma_o'])+lssfr_med,
                 x_linspace,
                 color='g', label='gaussian old')

        ax = plt.gca()

        ax.tick_params(axis='both',
                       direction='in',
                       length=10, width=3,
                       labelsize=20,
                       which='both',
                       top=True, right=True)
        ax.set_ylim([self.floor, self.ceil])
        plt.xlabel('$LsSFR$', fontsize=20)
        plt.ylabel('$x_1$', fontsize=20)

        plt.legend(ncol=1, loc='upper left',
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)

        plt.title('Evolution of $x_1$ on $LsSFR$', fontsize=20)

        plt.show()

#   ###########################################################################
#   ################################ EVOLDOUBLE ###############################
#   ###########################################################################


class EvolDouble():
    ''' '''

    FREEPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']

#   ################################# SETTER ##################################

    def set_names(self, names):
        '''Permet les extractions sur les pandas
        names = {'lssfr_name':       'lssfr',
                 'stretch_name':     'salt2.X1',
                 'stretch_err_name': 'salt2.X1.err',
                 'lssfr_err_d_name': 'lssfr.err_down',
                 'lssfr_err_u_name': 'lssfr.err_up',
                 'py_name':          'p(prompt)'}'''
        self.names = names

    def set_data(self, pandas):
        '''Donne les pandas des stretch, lssfr, et Py '''
        self.py = pandas[self.names['py_name']]

        self.stretch = pandas[self.names['stretch_name']]
        self.stretch_err = pandas[self.names['stretch_err_name']]

        self.lssfr = pandas[self.names['lssfr_name']]
        self.lssfr_err_d = pandas[self.names['lssfr_err_d_name']]
        self.lssfr_err_u = pandas[self.names['lssfr_err_u_name']]

        self.floor = np.floor(np.min(self.stretch)-0.4)
        self.ceil = np.ceil(np.max(self.stretch)+0.4)

    def set_param(self, param):
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

#   ################################## FITTER #################################

    def gauss(self, x, dx, mu, sigma):
        '''Le modèle de distribution'''
        return scipy.stats.norm.pdf(x, mu, scale=np.sqrt(dx**2+sigma**2))

    def likelihood_y(self, x, dx, mu_1, sigma_1):
        '''La fonction décrivant le modèle des SNe jeunes'''
        return self.gauss(x, dx, mu_1, sigma_1)

    def likelihood_o(self, x, dx, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_1, sigma_1) \
         + (1-a)*self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_tot(self, py, x, dx, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return py*self.likelihood_y(x, dx, mu_1, sigma_1) \
         + (1-py)*self.likelihood_o(x, dx, a, mu_1, sigma_1, mu_2, sigma_2)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.py, self.stretch,
                                                    self.stretch_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2, sigma_2)))

    def minimize(self, a_guess=False,
                 mu_1_guess=False, sigma_1_guess=False,
                 mu_2_guess=False, sigma_2_guess=False):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m_tot = im.Minuit(self.loglikelihood,
                               a=a_guess, limit_a=(0, 1),
                               mu_1=mu_1_guess, sigma_1=sigma_1_guess,
                               mu_2=mu_2_guess, sigma_2=sigma_2_guess,
                               print_level=0, pedantic=False)

        self.m_tot.migrad()

        self.set_param([self.m_tot.values['a'], self.m_tot.values['mu_1'],
                        self.m_tot.values['sigma_1'], self.m_tot.values['mu_2'],
                        self.m_tot.values['sigma_2']])

#   ################################# PLOTTER #################################

    def plt_scatter(self):
        '''Trace le nuage de points et les fits'''
        dgmap = plt.cm.get_cmap('viridis')
        dg_colors = [dgmap(i) for i in (1-self.py)]

        plt.scatter(self.lssfr, self.stretch, marker='o', s=60,
                    color=dg_colors, label='SNe data')

        plt.errorbar(self.lssfr, self.stretch,
                     xerr=[self.lssfr_err_d, self.lssfr_err_u],
                     yerr=self.stretch_err,
                     ecolor='gray', alpha=.3,
                     ls='none', label=None)

        plt.plot([lssfr_med, lssfr_med],
                 [self.floor, self.ceil],
                 color='b', alpha=.5, linewidth=2.0)

        x_linspace = np.linspace(self.floor, self.ceil, 3000)

        plt.plot(self.likelihood_y(x_linspace, 0, self.param['mu_1'],
                                   self.param['sigma_1'])+lssfr_med,
                 x_linspace,
                 color='r', label='gaussian young')
        plt.plot(-self.likelihood_o(x_linspace, 0, self.param['a'],
                                    self.param['mu_1'], self.param['sigma_1'],
                                    self.param['mu_2'], self.param['sigma_2'])
                                    + lssfr_med,
                 x_linspace,
                 color='g', label='gaussian old')

        ax = plt.gca()

        ax.tick_params(axis='both',
                       direction='in',
                       length=10, width=3,
                       labelsize=20,
                       which='both',
                       top=True, right=True)

        ax.set_ylim([self.floor, self.ceil])

        plt.xlabel('$LsSFR$', fontsize=20)
        plt.ylabel('$x_1$', fontsize=20)

        plt.legend(ncol=1, loc='upper left',
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)

        plt.title('Evolution of $x_1$ on $LsSFR$', fontsize=20)

        plt.show()


#   ###########################################################################
#   ################################# MOCKEVOL ################################
#   ###########################################################################


class MockEvol():
    ''' '''

    FREEPARAMETERS = []

#   ################################# SETTER ##################################

    def set_values(self, dict_param):
        '''Sets the values used by the mock to compute the repartition of SNe
        dict_param = evolD.param
        or
        dict_param = {'a':       a_value,
                      'mu_1':    mu_1_value,
                      'sigma_1': sigma_1_value,
                      'mu_2':    mu_2_value,
                      'sigma_2': sigma_2_value}'''
        self.param = dict_param

    def set_data(self, z, npoints):
        '''Given a redshift and a number of data, returns ?'''
        self.z = z
        self.npoints_y = int(self.delta(z)*npoints)
        self.npoints_o = int(self.psi(z)*npoints)

#   ################################# EXTMOD ##################################

    def delta(self, z):
        '''Gives the fraction of young SNe Ia as a function of redshift;
        taken from https://arxiv.org/abs/1806.03849'''
        K = 0.87
        Phi = 2.8
        return (K**(-1)*(1+z)**(-Phi)+1)**(-1)

    def psi(self, z):
        '''Gives the fraction of old SNe Ia as a function of redshift;
        taken from https://arxiv.org/abs/1806.03849'''
        K = 0.87
        Phi = 2.8
        return (K*(1+z)**(Phi)+1)**(-1)

#   ################################# SOLVER ##################################

    def solver(self):
        '''Gives the stretch distribution of young and old'''
        x1_o_1 = np.random.normal(loc=self.param['mu_1'],
                                  scale=abs(self.param['sigma_1']),
                                  size=int((self.npoints_o+5)
                                           * self.param['a']))
        x1_o_2 = np.random.normal(loc=self.param['mu_2'],
                                  scale=abs(self.param['sigma_2']),
                                  size=int((self.npoints_o+5)
                                           * (1-self.param['a'])))

        self.x1_o = np.concatenate([x1_o_1, x1_o_2])
        np.random.shuffle(self.x1_o)

        self.x1_o = self.x1_o[:self.npoints_o]

        self.x1_y = np.random.normal(loc=self.param['mu_1'],
                                     scale=abs(self.param['sigma_1']),
                                     size=self.npoints_y)

        self.floor = np.floor(np.min(np.concatenate((self.x1_o, self.x1_y)))
                              - 0.4)
        self.ceil = np.ceil(np.max(np.concatenate((self.x1_o, self.x1_y)))
                            + 0.5)

#   ################################# PLOTTER #################################

    def plotter(self, nbins, range):
        '''Shows the histogram of the stretch distribution
        range = [min, max]'''
        dgmap = plt.cm.get_cmap('viridis')

        plt.hist(self.x1_o,
                 bins=nbins, range=range,
                 histtype='step', lw=2,
                 color=dgmap(256), label='old SNe')

        plt.hist(self.x1_y,
                 bins=nbins, range=range,
                 histtype='step', lw=2,
                 color=dgmap(1), label='young SNe')

        ax = plt.gca()

        ax.tick_params(axis='both',
                       direction='in',
                       length=10, width=3,
                       labelsize=20,
                       which='both',
                       top=True, right=True)

        ax.set_xlim([self.floor, self.ceil])

        plt.xlabel('$x_1$', fontsize=20)
        plt.ylabel('#SNe Ia', fontsize=20)

        plt.legend(ncol=1, loc='upper left',
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)

        plt.title('Stretch distribution at $z = $' + str(self.z), fontsize=20)
