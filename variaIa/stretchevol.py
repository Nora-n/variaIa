#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import scipy
import iminuit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lssfr_med = -10.82

res_SNF = {'a': 0.47715121395809157,
           'mu_1': 0.3868646937414961,
           'sigma_1': 0.555465859117784,
           'mu_2': -1.5209713725396492,
           'sigma_2': 0.5187834441554329}

snf_a = res_SNF['a']
snf_mu_1 = res_SNF['mu_1']
snf_sigma_1 = res_SNF['sigma_1']
snf_mu_2 = res_SNF['mu_2']
snf_sigma_2 = res_SNF['sigma_2']


# =========================================================================== #
#                                                                             #
#                                 PRELIMINARY                                 #
#                                                                             #
# =========================================================================== #

def make_method(obj):
    """Decorator to make the function a method of *obj*.

    In the current context::
      @make_method(Axes)
      def toto(ax, ...):
          ...
    makes *toto* a method of `Axes`, so that one can directly use::
      ax.toto()
    COPYRIGHT: from Yannick Copin
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f

    return decorate


# =========================================================================== #
#                                                                             #
#                             CLASSIC STRETCHDIST                             #
#                                                                             #
# =========================================================================== #


class StretchDist():
    '''USAGE :
    evol = stretchevol.EvolCHOICE()

    evol.set_data(dataframe)

    evol.minimize()

    evol.plotter()
    '''

    def set_data(self, pandas, py=True):
        '''Pour une meilleure utilisation des données'''
        self.pd = pandas

        self.redshifts = pandas.redshifts
        self.stretchs = pandas.stretchs
        self.stretchs_err = pandas.stretchs_err

        if py:
            self.info = pandas.infor.values
        else:
            self.info = self.delta(pandas.redshifts)

        self.py = pandas.py
        self.lssfr = pandas.lssfr
        self.lssfr_err_d = pandas.lssfr_err_d
        self.lssfr_err_u = pandas.lssfr_err_u

        self.floor = np.floor(np.min(self.stretchs)-0.4)
        self.ceil = np.ceil(np.max(self.stretchs)+0.3)


# =========================================================================== #
#                                                                             #
#                                    MODELS                                   #
#                                                                             #
# =========================================================================== #

# =========================================================================== #
#                                 EvolHowell                                  #
# =========================================================================== #


class Evol2G2M2S(StretchDist):
    '''Howell'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + '=' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                           BaseModel Struc                           #
    # =================================================================== #

    def __new__(cls, *arg, **kwargs):
        '''Upgrade of the New function to enable the _minuit_ black magic'''
        obj = super(Evol2G2M2S, cls).__new__(cls)

        # ----------------------------------------------------------- #
        #                          Probalizer                         #
        # ----------------------------------------------------------- #

        exec("@make_method(Evol2G2M2S)\n" +
             "def logprob(self, %s):\n" % (", ".join(obj.FREEPARAMETERS)) +
             "    loglikelihood = self.loglikelihood(%s)\n" %
             (", ".join(obj.FREEPARAMETERS)) +
             "    logprior = self.logprior()\n" +
             "    return logprior + loglikelihood")

        # ----------------------------------------------------------- #
        #                          Minimizer                          #
        # ----------------------------------------------------------- #

        exec("@make_method(Evol2G2M2S)\n" +
             "def minimize(self, print_level=0, **kwargs):\n" +
             "   '''Renvoie la meilleure valeur des paramètres'''\n" +
             "   self.m_tot = iminuit.Minuit(self.logprob,\n" +
             "                               print_level=print_level,\n" +
             "                               pedantic=False,\n" +
             "                               **kwargs,\n" +
             "                               %s)\n"
             % (", \n                               ".join(obj.GUESS)) +
             "\n" +
             "   self.migrad_out = self.m_tot.migrad()\n" +
             "\n" +
             "   self.set_param([self.m_tot.values[k] for k in " +
             "self.FREEPARAMETERS])\n")

        if len(obj.YOUNGPARAMETERS) == 0:
            return obj

        # ----------------------------------------------------------- #
        #                           Plotter                           #
        # ----------------------------------------------------------- #

        exec("@make_method(Evol2G2M2S)\n" +
             "def plotter(self, name=None):\n" +
             "   '''Trace les fits et les données si lssfr donné'''\n" +
             "   dgmap = plt.cm.get_cmap('viridis')\n" +
             "\n" +
             "   fig = plt.figure(figsize=[8, 5])\n" +
             "   ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])\n" +
             "\n" +
             "   if hasattr(self, 'lssfr'):\n" +
             "       dg_colors = [dgmap(i) for i in (1-self.py)]\n" +
             "       ax.scatter(self.lssfr, self.stretch, marker='o',\n" +
             "                  s=100, linewidths=0.5,\n" +
             "                  facecolors=dg_colors, edgecolors='0.7',\n" +
             "                  label='SNe data', zorder=8)\n" +
             "\n" +
             "       ax.errorbar(self.lssfr, self.stretch,\n" +
             "                   xerr=[self.lssfr_err_d, self.lssfr_err_u],\n"
             +
             "                   yerr=self.stretch_err,\n" +
             "                   ecolor='0.7', alpha=1, ms=0,\n" +
             "                   ls='none', label=None, zorder=5)\n" +
             "\n" +
             "   ax.vline(lssfr_med,\n" +
             "            color='0.7', alpha=.5, linewidth=2.0)\n" +
             "\n" +
             "   x_linspace = np.linspace(self.floor, self.ceil, 3000)\n" +
             "\n" +
             "   plt.fill_betweenx(x_linspace,\n" +
             "                     2*self.likelihood_y(x_linspace, 0, \n" +
             "                                         %s)\n"
             % (", \n                                         ".join(obj.PLTYOUNG)) +
             "                     + lssfr_med,\n" +
             "                     lssfr_med,\n" +
             "                     facecolor=plt.cm.viridis(0.05, 0.1),\n"
             +
             "                     edgecolor=plt.cm.viridis(0.05, 0.8),\n"
             +
             "                     lw=2, label='model young')\n"
             "\n" +
             "   plt.fill_betweenx(x_linspace,\n" +
             "                     -3*self.likelihood_o(x_linspace, 0, \n" +
             "                                         %s)\n"
             % (", \n                                         ".join(obj.PLTOLD)) +
             "                     + lssfr_med,\n" +
             "                     lssfr_med,\n" +
             "                     facecolor=plt.cm.viridis(0.95, 0.1),\n"
             +
             "                     edgecolor=plt.cm.viridis(0.95, 0.8),\n"
             +
             "                     lw=2, label='model old')\n" +
             "\n" +
             "   ax.tick_params(direction='in',\n" +
             "                  length=5, width=1,\n" +
             "                  labelsize=15,\n" +
             "                  top=True, right=True)\n" +
             "\n" +
             "   ax.set_ylim([self.floor, self.ceil])\n" +
             "\n" +
             "   ax.set_xlabel(r'$\mathrm{log(LsSFR)}$',fontsize='x-large')\n" +
             "   ax.set_ylabel(r'$\mathrm{x}_1$', fontsize='x-large')\n" +
             "\n" +
             "   plt.legend(ncol=1, loc='upper left')\n" +
             "\n" +
             "   plt.title(name, fontsize='x-large')")

        return obj

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

    def set_param(self, param):
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

    # ------------------------------------------------------------------- #
    #                               EXTFUNC                               #
    # ------------------------------------------------------------------- #

    def delta(self, z):
        '''Gives the fraction of young SNe Ia as a function of redshift;
        taken from https://arxiv.org/abs/1806.03849'''
        K = 0.87
        Phi = 2.8
        return (K**(-1)*(1+z)**(-Phi)+1)**(-1)

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

#    def gauss(self, x, dx, mu, sigma):
#        '''Le modèle de distribution'''
#        return scipy.stats.norm.pdf(x, mu, scale=np.sqrt(dx**2+sigma**2))

    def gauss(self, x, dx, mu, sigma, normed=True):
        '''Le modèle de distribution'''
        sigma_eff = np.sqrt(dx**2+sigma**2)
        norm = 1 if normed else np.sqrt(2*np.pi)*sigma_eff
        return norm*scipy.stats.norm.pdf(x, mu, scale=sigma_eff)

    def likelihood_y(self, x, dx, mu_1, sigma_1):
        '''La fonction décrivant le modèle des SNe jeunes'''
        return self.gauss(x, dx, mu_1, sigma_1)

    def likelihood_o(self, x, dx, mu_2, sigma_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_tot(self, info, x, dx, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, mu_2, sigma_2)

    def loglikelihood(self, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2)))

    def logprior(self):
        '''Impose des contraintes sur les paramètres'''
        return 0

    def get_logl(self, **kwargs):
        if not hasattr(self, 'migrad_out'):
            self.minimize(**kwargs)
        return self.migrad_out[0]['fval']

    def get_bic(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return mdlogl + k*np.log(len(self.stretchs))

    def get_aicc(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return 2*k + mdlogl + (2*k*(k+1))/(len(self.stretchs)-k-1)

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

    def scatter(self, model=True, ax=None, show_leg=True,
                mod_lw=2, lw=.5, elw=1, ealpha=1, s=80,
                facealpha=1, fontsize='large'):
        '''Trace le nuage de points et les fits
        model=False ne montre que les données du pandas'''
        dgmap = plt.cm.get_cmap('viridis')
        dg_colors = [dgmap(i, facealpha) for i in (1-self.py)]

        if ax is None:
            fig = plt.figure(figsize=[8, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        if hasattr(self, 'lssfr'):
            ax.scatter(self.lssfr, self.stretchs, marker='o',
                       s=s, linewidths=lw,
                       facecolors=dg_colors, edgecolors="0.7",
                       zorder=8)

            ax.errorbar(self.lssfr, self.stretchs,
                        xerr=[self.lssfr_err_d, self.lssfr_err_u],
                        yerr=self.stretchs_err,
                        ecolor='0.7', alpha=ealpha, ms=0,
                        lw=elw,
                        ls='none', label=None, zorder=5)

        ax.vline(lssfr_med,
                 color='0', alpha=1, linewidth=lw)

        if model is True:
            self.show_model(ax=ax, shift=lssfr_med, lw=mod_lw)

        ax.set_ylim([self.floor, self.ceil])

        ax.set_xlabel(r'$\mathrm{log(LsSFR)}$', fontsize=fontsize)
        ax.set_ylabel(r'$\mathrm{x}_1$', fontsize=fontsize)

        if show_leg:
            ax.legend(ncol=1, loc='upper left')

        # plt.title('1GSNF model', fontsize=20)

    def show_model(self, ax=None, shift=0,
                   facealpha=0.1, edgealpha=.8, **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[8, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        x_linspace = np.linspace(self.floor, self.ceil, 3000)

        ax.fill_betweenx(x_linspace,
                         2*self.likelihood_y(x_linspace, 0,
                                             self.param['mu_1'],
                                             self.param['sigma_1'])
                         + shift,
                         shift,
                         facecolor=plt.cm.viridis(0.05, facealpha),
                         edgecolor=plt.cm.viridis(0.05, edgealpha),
                         label='model young', **kwargs)

        ax.fill_betweenx(x_linspace,
                         -3*self.likelihood_o(x_linspace, 0,
                                              self.param['aa'],
                                              self.param['mu_1'],
                                              self.param['sigma_1'],
                                              self.param['mu_2'],
                                              self.param['sigma_2'])
                         + shift,
                         shift,
                         facecolor=plt.cm.viridis(0.95, facealpha),
                         edgecolor=plt.cm.viridis(0.95, edgealpha),
                         label='model old', **kwargs)

# =========================================================================== #
#                                 EvolHowellF                                 #
# =========================================================================== #


class Evol2G2M2SF(Evol2G2M2S):
    '''Howell:$f$'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, f, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return f*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-f)*self.likelihood_o(x, dx, mu_2, sigma_2)

    def loglikelihood(self, f, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    f,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2)))

# =========================================================================== #
#                                 EvolSimple                                  #
# =========================================================================== #


class Evol1G1M1S(Evol2G2M2S):
    '''Gaussian'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigma']
    GUESSVAL = [0, 1]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, mu, sigma):
        '''La fonction de distribution'''
        return self.gauss(x, dx, mu, sigma)

    def loglikelihood(self, mu, sigma):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    mu, sigma)))

# =========================================================================== #
#                                 EvolKessler                                 #
# =========================================================================== #


class Evol1G1M2S(Evol2G2M2S):
    '''Asymetric'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigma_m', 'sigma_p']
    GUESSVAL = [0.973, 1.5, 0.5]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, mu, sigma_m, sigma_p):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        flag_up = x >= mu
        likelihood = np.zeros(len(x))

        likelihood[flag_up] = self.gauss(x[flag_up], dx[flag_up],
                                         mu, sigma_p, normed=False)
        likelihood[~flag_up] = self.gauss(x[~flag_up], dx[~flag_up],
                                          mu, sigma_m, normed=False)
        norm = np.sqrt(2*np.pi)*(0.5*np.sqrt(dx**2+sigma_m**2)
                                 + 0.5*np.sqrt(dx**2+sigma_p**2))
        return likelihood/norm

    def loglikelihood(self, mu, sigma_m, sigma_p):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    mu,
                                                    sigma_m, sigma_p)))

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

    def traceur(self, model=True):
        '''Trace le nuage de points et les fits
        model=False ne montre que les données du pandas'''
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        x_linspace = np.linspace(self.floor, self.ceil, 3000)

        flag_up = x_linspace >= self.param['mu']

        if model is True:
            ax.fill_between(x_linspace[flag_up],
                            (self.param['sigma_p']) * np.sqrt(np.pi/2) *
                            self.gauss(x_linspace[flag_up],
                                       np.zeros(len(x_linspace[flag_up])),
                                       self.param['mu'],
                                       self.param['sigma_p']),
                            facecolor=plt.cm.viridis(0.05, 0.1),
                            edgecolor=plt.cm.viridis(0.05, 0.8),
                            lw=2)

            ax.fill_between(x_linspace[~flag_up],
                            (self.param['sigma_m']) * np.sqrt(np.pi/2) *
                            self.gauss(x_linspace[~flag_up],
                                       np.zeros(len(x_linspace[~flag_up])),
                                       self.param['mu'],
                                       self.param['sigma_m']),
                            facecolor=plt.cm.viridis(0.05, 0.1),
                            edgecolor=plt.cm.viridis(0.05, 0.8),
                            lw=2)

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=15,
                       top=True, right=True)

        ax.set_xlim([self.floor, self.ceil])
        # ax.set_ylim([-11, -9])

        ax.set_ylabel(r'$\mathrm{Probability}$', fontsize='x-large')
        ax.set_xlabel(r'$\mathrm{x}_1$', fontsize='x-large')

# =========================================================================== #
#                                   EvolNR1S                                  #
# =========================================================================== #


class Evol3G2M1S(Evol2G2M2S):
    '''Base$-(\sigma_2)$'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_o(self, x, dx, a, mu_1, sigma_1, mu_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_1, sigma_1) + \
            (1-a)*self.gauss(x, dx, mu_2, sigma_1)

    def likelihood_tot(self, info, x, dx, a, mu_1, sigma_1, mu_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, a, mu_1, sigma_1, mu_2)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2)))

# =========================================================================== #
#                                  EvolNR1SNF                                 #
# =========================================================================== #


class Evol3G2M1SSNF(Evol3G2M1S):
    '''Base$-(\sigma_2)$ on SNf'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, py, x, dx, a, mu_1, sigma_1, mu_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return py*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-py)*self.likelihood_o(x, dx, a, mu_1, sigma_1, mu_2)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.py,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2)))

# =========================================================================== #
#                                   EvolNR1S                                  #
# =========================================================================== #


class Evol3G2M1SF(Evol3G2M1S):
    '''Base$-(\sigma_2):f$'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2']
    FREEPARAMETERS = np.append(GLOBALPARAMETERS, OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, f, a, mu_1, sigma_1, mu_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return f*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-f)*self.likelihood_o(x, dx, a, mu_1, sigma_1, mu_2)

    def loglikelihood(self, f, a, mu_1, sigma_1, mu_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    f, a,
                                                    mu_1, sigma_1,
                                                    mu_2)))

# =========================================================================== #
#                                   EvolNR2S                                  #
# =========================================================================== #


class Evol3G2M2S(Evol2G2M2S):
    '''Base'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['aa', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def get_a(self, aa):
        '''Get a in [0, 1] from aa parameter'''
        return(np.arctan(aa)/np.pi + 0.5)

    def likelihood_o(self, x, dx, aa, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return self.get_a(aa)*self.gauss(x, dx, mu_1, sigma_1) + \
            (1-self.get_a(aa))*self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_tot(self, info, x, dx, aa, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, aa, mu_1, sigma_1,
                                       mu_2, sigma_2)

    def loglikelihood(self, aa, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    aa, mu_1, sigma_1,
                                                    mu_2, sigma_2)))

# =========================================================================== #
#                                  EvolNR2SF                                  #
# =========================================================================== #


class Evol3G2M2SF(Evol3G2M2S):
    '''Base:$f$'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(GLOBALPARAMETERS, OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, f, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return f*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-f)*self.likelihood_o(x, dx, a,
                                    mu_1, sigma_1,
                                    mu_2, sigma_2)

    def loglikelihood(self, f, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    f, a,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2)))

# =========================================================================== #
#                                  EvolNR2SNF                                 #
# =========================================================================== #


class Evol3G2M2SSNF(Evol3G2M2S):
    '''Base on SNf'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['aa', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, py, x, dx, aa, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return py*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-py)*self.likelihood_o(x, dx, aa, mu_1, sigma_1, mu_2, sigma_2)

    def loglikelihood(self, aa, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.py,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    aa,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2)))

# =========================================================================== #
#                                  EvolNRTOT                                  #
# =========================================================================== #


class Evol3G3M3S(Evol3G2M2S):
    r'''Base$+(\mu_1^{\mathrm{O}}, \sigma_1^{\mathrm{O}})$'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_2', 'sigma_2', 'mu_3', 'sigma_3']
    FREEPARAMETERS = ['a',
                      'mu_1', 'sigma_1',
                      'mu_2', 'sigma_2',
                      'mu_3', 'sigma_3']
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_o(self, x, dx, a, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_2, sigma_2) + \
            (1-a)*self.gauss(x, dx, mu_3, sigma_3)

    def likelihood_tot(self, info, x, dx, a,
                       mu_1, sigma_1,
                       mu_2, sigma_2,
                       mu_3, sigma_3):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, a,
                                       mu_2, sigma_2,
                                       mu_3, sigma_3)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    a,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2,
                                                    mu_3, sigma_3)))

# =========================================================================== #
#                                  EvolNRTOTF                                 #
# =========================================================================== #


class Evol3G3M3SF(Evol3G3M3S):
    r'''Base$+(\mu_1^{\mathrm{O}}, \sigma_1^{\mathrm{O}}):f$'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_2', 'sigma_2', 'mu_3', 'sigma_3']
    FREEPARAMETERS = ['f', 'a',
                      'mu_1', 'sigma_1',
                      'mu_2', 'sigma_2',
                      'mu_3', 'sigma_3']
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = [k + ' = ' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, f, a,
                       mu_1, sigma_1,
                       mu_2, sigma_2,
                       mu_3, sigma_3):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return f*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-f)*self.likelihood_o(x, dx, a,
                                    mu_2, sigma_2,
                                    mu_3, sigma_3)

    def loglikelihood(self, f, a, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    f, a,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2,
                                                    mu_3, sigma_3)))

# =========================================================================== #
#                                                                             #
#                                   MOCKEVOL                                  #
#                                                                             #
# =========================================================================== #


class MockEvol():
    ''' '''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    FREEPARAMETERS = []

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

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
        '''Given a redshift and a number of data, returns the number of
        young and old SNe'''
        self.z = z
        self.npoints_y = int(self.delta(z)*npoints)
        self.npoints_o = int(self.psi(z)*npoints)

    # ------------------------------------------------------------------- #
    #                               EXTFUNC                               #
    # ------------------------------------------------------------------- #

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

    # ------------------------------------------------------------------- #
    #                               SOLVER                                #
    # ------------------------------------------------------------------- #

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

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

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

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=15,
                       top=True, right=True)

        ax.set_xlim([self.floor, self.ceil])

        plt.xlabel('$x_1$', fontsize=20)
        plt.ylabel('#SNe Ia', fontsize=20)

        plt.legend(ncol=1, loc='upper left',
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)

        plt.title('Stretch distribution at $z = $' + str(self.z), fontsize=20)


# =========================================================================== #
#                                                                             #
#                                  COMPARISON                                 #
#                                                                             #
# =========================================================================== #

    # ------------------------------------------------------------------- #
    #                               EXTFUNC                               #
    # ------------------------------------------------------------------- #

def get_proba(best, model):
    return np.exp((best.get_aicc() - model.get_aicc())/2)

    # ------------------------------------------------------------------- #
    #                               SOLVER                                #
    # ------------------------------------------------------------------- #


def zmax_impact(zmax):
    d = pd.read_csv('../Data/data_cheat.csv', sep=' ', index_col='CID')
    d_snf = pd.read_csv('../Data/lssfr_paper_full_sntable.csv', sep=',')

    surv = {'SNF':  d_snf.loc[d_snf['name'].str.contains('SNF|LSQ|PTF',
                                                         na=False,
                                                         regex=True)],
            'SDSS': d[d['IDSURVEY'] == 1],
            'PS1':  d[d['IDSURVEY'] == 15],
            'SNLS': d[d['IDSURVEY'] == 4],
            'HST':  d[d['IDSURVEY'].isin([101, 100, 106])]}

    surveys = list(zmax.keys())

    zmax_cuts = dict()
    z_zcuts = dict()
    x1_zcuts = dict()
    x1_err_zcuts = dict()

    evol1G1M1S = Evol1G1M1S()
    evol1G1M2S = Evol1G1M2S()
    evol2G2M2S = Evol2G2M2S()
    evol2G2M2SF = Evol2G2M2SF()
    evol3G2M1S = Evol3G2M1S()
    evol3G2M1SF = Evol3G2M1SF()
    evol3G2M2S = Evol3G2M2S()
    evol3G2M2SF = Evol3G2M2SF()
    evol3G3M3S = Evol3G3M3S()
    evol3G3M3SF = Evol3G3M3SF()

    d_mod_comp = []
    NR_params = []

    # gen = (survey for survey in surveys if survey != 'SNF')

    for i in range(len(zmax['SDSS'])):
        for survey in surveys[1:]:
            zmax_cuts[survey] = np.where(surv[survey].zCMB.values
                                         < zmax[survey][i])
            z_zcuts[survey] = surv[survey].zCMB.values[zmax_cuts[survey]]
            x1_zcuts[survey] = surv[survey].x1.values[zmax_cuts[survey]]
            x1_err_zcuts[survey] = surv[survey].x1ERR.values[zmax_cuts[survey]]

        zmax_cuts['SNF'] = np.where(surv['SNF']['host.zcmb'].values
                                    < zmax['SNF'][i])
        z_zcuts['SNF'] = surv['SNF']['host.zcmb'].values[zmax_cuts['SNF']]
        x1_zcuts['SNF'] = surv['SNF']['salt2.X1'].values[zmax_cuts['SNF']]
        x1_err_zcuts['SNF'] = surv['SNF']['salt2.X1.err'].values[zmax_cuts['SNF']]

        datax_all = np.concatenate(
            (np.concatenate(
                (np.concatenate(
                    (np.concatenate((x1_zcuts['SNF'],
                                     x1_zcuts['SDSS'])),
                     x1_zcuts['PS1'])),
                 x1_zcuts['SNLS'])),
             x1_zcuts['HST']))

        datax_err_all = np.concatenate(
            (np.concatenate(
                (np.concatenate(
                    (np.concatenate((x1_err_zcuts['SNF'],
                                     x1_err_zcuts['SDSS'])),
                     x1_err_zcuts['PS1'])),
                 x1_err_zcuts['SNLS'])),
             x1_err_zcuts['HST']))

        dataz_all = np.concatenate(
            (np.concatenate(
                (np.concatenate(
                    (np.concatenate((z_zcuts['SNF'],
                                     z_zcuts['SDSS'])),
                     z_zcuts['PS1'])),
                 z_zcuts['SNLS'])),
             z_zcuts['HST']))

        frame = pd.DataFrame({'Name': [],
                              'ln L': [],
                              'AICc': [],
                              '$\Delta$ AICc': [],
                              'Proba': []})

        del(evol3G2M1S)
        evol3G2M1S = Evol3G2M1S()
        evol3G2M1S.set_data(dataz_all, datax_all, datax_err_all)
        evol3G2M1S.minimize()

        frame.loc[0, 'Name'] = type(evol3G2M1S).__name__
        frame.loc[0, 'ln L'] = evol3G2M1S.get_logl()
        frame.loc[0, 'AICc'] = evol3G2M1S.get_aicc()
        frame.loc[0, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol3G2M1S.get_aicc())
        frame.loc[0, 'Proba'] = get_proba(evol3G2M1S,
                                          evol3G2M1S)

        del(evol3G2M2S)
        evol3G2M2S = Evol3G2M2S()
        evol3G2M2S.set_data(dataz_all, datax_all, datax_err_all)
        evol3G2M2S.minimize()

        frame.loc[1, 'Name'] = type(evol3G2M2S).__name__
        frame.loc[1, 'ln L'] = evol3G2M2S.get_logl()
        frame.loc[1, 'AICc'] = evol3G2M2S.get_aicc()
        frame.loc[1, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol3G2M2S.get_aicc())
        frame.loc[1, 'Proba'] = get_proba(evol3G2M1S,
                                          evol3G2M2S)

        del(evol2G2M2S)
        evol2G2M2S = Evol2G2M2S()
        evol2G2M2S.set_data(dataz_all, datax_all, datax_err_all)
        evol2G2M2S.minimize()

        frame.loc[2, 'Name'] = type(evol2G2M2S).__name__
        frame.loc[2, 'ln L'] = evol2G2M2S.get_logl()
        frame.loc[2, 'AICc'] = evol2G2M2S.get_aicc()
        frame.loc[2, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol2G2M2S.get_aicc())
        frame.loc[2, 'Proba'] = get_proba(evol3G2M1S,
                                          evol2G2M2S)

        del(evol3G3M3S)
        evol3G3M3S = Evol3G3M3S()
        evol3G3M3S.set_data(dataz_all, datax_all, datax_err_all)
        evol3G3M3S.minimize()

        frame.loc[3, 'Name'] = type(evol3G3M3S).__name__
        frame.loc[3, 'ln L'] = evol3G3M3S.get_logl()
        frame.loc[3, 'AICc'] = evol3G3M3S.get_aicc()
        frame.loc[3, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol3G3M3S.get_aicc())
        frame.loc[3, 'Proba'] = get_proba(evol3G2M1S,
                                          evol3G3M3S)

        del(evol3G3M3SF)
        evol3G3M3SF = Evol3G3M3SF()
        evol3G3M3SF.set_data(dataz_all, datax_all, datax_err_all)
        evol3G3M3SF.minimize(limit_f=(0, 1), limit_a=(0, 1))

        frame.loc[4, 'Name'] = type(evol3G3M3SF).__name__
        frame.loc[4, 'ln L'] = evol3G3M3SF.get_logl()
        frame.loc[4, 'AICc'] = evol3G3M3SF.get_aicc()
        frame.loc[4, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol3G3M3SF.get_aicc())
        frame.loc[4, 'Proba'] = get_proba(evol3G2M1S,
                                          evol3G3M3SF)

        del(evol3G2M2SF)
        evol3G2M2SF = Evol3G2M2SF()
        evol3G2M2SF.set_data(dataz_all, datax_all, datax_err_all)
        evol3G2M2SF.minimize(limit_f=(0, 1), limit_a=(0, 1))

        frame.loc[5, 'Name'] = type(evol3G2M2SF).__name__
        frame.loc[5, 'ln L'] = evol3G2M2SF.get_logl()
        frame.loc[5, 'AICc'] = evol3G2M2SF.get_aicc()
        frame.loc[5, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol3G2M2SF.get_aicc())
        frame.loc[5, 'Proba'] = get_proba(evol3G2M1S,
                                          evol3G2M2SF)

        del(evol2G2M2SF)
        evol2G2M2SF = Evol2G2M2SF()
        evol2G2M2SF.set_data(dataz_all, datax_all, datax_err_all)
        evol2G2M2SF.minimize()

        frame.loc[6, 'Name'] = type(evol2G2M2SF).__name__
        frame.loc[6, 'ln L'] = evol2G2M2SF.get_logl()
        frame.loc[6, 'AICc'] = evol2G2M2SF.get_aicc()
        frame.loc[6, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol2G2M2SF.get_aicc())
        frame.loc[6, 'Proba'] = get_proba(evol3G2M1S,
                                          evol2G2M2SF)

        del(evol3G2M1SF)
        evol3G2M1SF = Evol3G2M1SF()
        evol3G2M1SF.set_data(dataz_all, datax_all, datax_err_all)
        evol3G2M1SF.minimize(limit_f=(0, 1), limit_a=(0, 1))

        frame.loc[7, 'Name'] = type(evol3G2M1SF).__name__
        frame.loc[7, 'ln L'] = evol3G2M1SF.get_logl()
        frame.loc[7, 'AICc'] = evol3G2M1SF.get_aicc()
        frame.loc[7, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol3G2M1SF.get_aicc())
        frame.loc[7, 'Proba'] = get_proba(evol3G2M1S,
                                          evol3G2M1SF)

        del(evol1G1M2S)
        evol1G1M2S = Evol1G1M2S()
        evol1G1M2S.set_data(dataz_all, datax_all, datax_err_all)
        evol1G1M2S.minimize()

        frame.loc[8, 'Name'] = type(evol1G1M2S).__name__
        frame.loc[8, 'ln L'] = evol1G1M2S.get_logl()
        frame.loc[8, 'AICc'] = evol1G1M2S.get_aicc()
        frame.loc[8, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol1G1M2S.get_aicc())
        frame.loc[8, 'Proba'] = get_proba(evol3G2M1S,
                                          evol1G1M2S)

        del(evol1G1M1S)
        evol1G1M1S = Evol1G1M1S()
        evol1G1M1S.set_data(dataz_all, datax_all, datax_err_all)
        evol1G1M1S.minimize()

        frame.loc[9, 'Name'] = type(evol1G1M1S).__name__
        frame.loc[9, 'ln L'] = evol1G1M1S.get_logl()
        frame.loc[9, 'AICc'] = evol1G1M1S.get_aicc()
        frame.loc[9, '$\Delta$ AICc'] = (evol3G2M1S.get_aicc()
                                         - evol1G1M1S.get_aicc())
        frame.loc[9, 'Proba'] = get_proba(evol3G2M1S,
                                          evol1G1M1S)

        d_mod_comp.append(frame)

        NR_params.append([round(evol3G2M2S.param['a'], 4),
                          round(evol3G2M2S.param['mu_1'], 4),
                          round(evol3G2M2S.param['mu_2'], 4)])

    return(d_mod_comp, NR_params)


# =========================================================================== #
#                                                                             #
#                                  GENERIQUE                                  #
#                                                                             #
# =========================================================================== #


class fitter(StretchDist):
    ''' '''

    def set_model(self, classname):
        self.model = getattr(sys.modules[__name__], classname)

    def fit(self):
        ''' '''
        model = self.model()
        model.set_data(self.pd)
        model.minimize()
        self.model = model
