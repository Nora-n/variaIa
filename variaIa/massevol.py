#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import scipy
import iminuit
import numpy as np
import matplotlib.pyplot as plt

lssfr_med = -10.82

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
#                                  GENERIQUE                                  #
#                                                                             #
# =========================================================================== #


class generic():
    '''Usage:
       gen = massevol.generic()
       gen.set_model('model')
       fitted_model = gen.fit(pandas)

       fitted_model.param'''

    def set_model(self, classname):
        self.model = getattr(sys.modules[__name__], classname)

    def fit(self, pandas, **kwargs):
        ''' '''
        model = self.model(pandas)
        model.minimize(**kwargs)
        return(model)

# =========================================================================== #
#                                                                             #
#                                    MODELS                                   #
#                                                                             #
# =========================================================================== #

# =========================================================================== #
#                              Asymgauss + gauss                              #
# =========================================================================== #


class Evol3G3M4S():
    '''3G3M4S'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_2', 'sigma_2', 'mu', 'sigmadown', 'sigmaup']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    PLTALL = ["self.param['" + k + "']" for k in FREEPARAMETERS]
    GUESSVAL = [9.5, 2, 0.5, 9, 0.5, 10.5, 1, 0.5]
    GUESS = [k + '=' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Initial                               #
    # =================================================================== #

    def __init__(self, pandas, py=True):
        '''Pour une meilleure utilisation des données'''
        self.pd = pandas

        self.redshifts = pandas.redshifts
        self.hostmass = pandas.hostmass
        self.hostmass_err = pandas.hostmass_err

        if py:
            self.info = pandas.infor.values
        else:
            self.info = self.delta(pandas.redshifts)

        self.py = pandas.py
        self.lssfr = pandas.lssfr
        self.lssfr_err_d = pandas.lssfr_err_d
        self.lssfr_err_u = pandas.lssfr_err_u

        self.floor = np.floor(np.min(self.hostmass[self.lssfr != 0]))
        self.ceil = np.ceil(np.max(self.hostmass))

    # =================================================================== #
    #                           BaseModel Struc                           #
    # =================================================================== #

    def __new__(cls, *arg, **kwargs):
        '''Upgrade of the New function to enable the _minuit_ black magic'''
        obj = super(Evol3G3M4S, cls).__new__(cls)

        # ----------------------------------------------------------- #
        #                          Probalizer                         #
        # ----------------------------------------------------------- #

        exec("@make_method(Evol3G3M4S)\n" +
             "def logprob(self, %s):\n" % (", ".join(obj.FREEPARAMETERS)) +
             "    loglikelihood = self.loglikelihood(%s)\n" %
             (", ".join(obj.FREEPARAMETERS)) +
             "    logprior = self.logprior()\n" +
             "    return logprior + loglikelihood")

        # ----------------------------------------------------------- #
        #                          Minimizer                          #
        # ----------------------------------------------------------- #

        exec("@make_method(Evol3G3M4S)\n" +
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

        # ----------------------------------------------------------- #
        #                           Plotter                           #
        # ----------------------------------------------------------- #

        exec("@make_method(Evol3G3M4S)\n" +
             "def plot_a(self, x_lin):\n" +
             "    return self.likelihood_tot(self.info, \n" +
             "                               x_lin, \n" +
             "                               np.zeros(len(x_lin)), \n" +
             "                               %s)\n"
             % (",\n                               ".join(obj.PLTALL)))

        exec("@make_method(Evol3G3M4S)\n" +
             "def plot_y(self, x_lin):\n" +
             "    return self.likelihood_y(x_lin, \n" +
             "                             np.zeros(len(x_lin)), \n" +
             "                            %s)\n"
             % (",\n                            ".join(obj.PLTYOUNG)))

        exec("@make_method(Evol3G3M4S)\n" +
             "def plot_o(self, x_lin):\n" +
             "    return self.likelihood_o(x_lin, \n" +
             "                             np.zeros(len(x_lin)), \n" +
             "                            %s)\n"
             % (",\n                            ".join(obj.PLTOLD)))

        return obj

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               EXTFUNC                               #
    # ------------------------------------------------------------------- #

    @staticmethod
    def delta(z):
        '''Gives the fraction of young SNe Ia as a function of redshift;
        taken from https://arxiv.org/abs/1806.03849'''
        K = 0.87
        Phi = 2.8
        return (K**(-1)*(1+z)**(-Phi)+1)**(-1)

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

    def set_param(self, param):
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def gauss(self, x, dx, mu, sigma, normed=True):
        '''Le modèle de distribution'''
        sigma_eff = np.sqrt(dx**2+sigma**2)
        norm = 1 if normed else np.sqrt(2*np.pi)*sigma_eff
        return norm*scipy.stats.norm.pdf(x, mu, scale=sigma_eff)

    def asymgauss(self, x, dx, mu, sigmadown, sigmaup):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        flag_up = x >= mu
        res = np.zeros(len(x))
        res[flag_up] = self.gauss(x[flag_up], dx[flag_up],
                                  mu, sigmaup, normed=False)
        res[~flag_up] = self.gauss(x[~flag_up], dx[~flag_up],
                                   mu, sigmadown, normed=False)
        norm = np.sqrt(2*np.pi)*(0.5*np.sqrt(dx**2+sigmadown**2)
                                 + 0.5*np.sqrt(dx**2+sigmaup**2))
        return res/norm

    def likelihood_y(self, x, dx, mu_1, sigma_1):
        '''La fonction décrivant le modèle des SNe jeunes'''
        return self.gauss(x, dx, mu_1, sigma_1)

    def likelihood_o(self, x, dx, a, mu_2, sigma_2,
                     mu, sigmadown, sigmaup):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_2, sigma_2) + \
            (1-a)*self.asymgauss(x, dx, mu, sigmadown, sigmaup)

    def likelihood_tot(self, info, x, dx, mu_1, sigma_1,
                       a, mu_2, sigma_2,
                       mu, sigmadown, sigmaup):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, a, mu_2, sigma_2,
                                       mu, sigmadown, sigmaup)

    def loglikelihood(self, mu_1, sigma_1,
                      a, mu_2, sigma_2,
                      mu, sigmadown, sigmaup):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.hostmass,
                                                    self.hostmass_err,
                                                    mu_1, sigma_1,
                                                    a, mu_2, sigma_2,
                                                    mu, sigmadown, sigmaup)))

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

        return mdlogl + k*np.log(len(self.hostmass))

    def get_aic(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return 2*k + mdlogl

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

        # ----------------------------------------------------------- #
        #                           Scatter                           #
        # ----------------------------------------------------------- #

    def scatter(self, model=True, ax=None, show_leg=False, lssfr=True,
                mod_lw=1, lw=1, elw=1, ealpha=1, s=80,
                facealpha=1, fontsize='large', yotext=True):
        '''Trace le nuage de points et les fits
        model=False ne montre que les données du pandas'''
        dgmap = plt.cm.get_cmap('viridis')

        if ax is None:
            fig = plt.figure(figsize=[7, 3.5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        if lssfr:
            where = self.pd['lssfr'] != 0
            dg_colors = [dgmap(i, facealpha) for i in (1-self.py[where])]
            ax.scatter(self.lssfr[where], self.hostmass[where],
                       marker='o',
                       s=s, linewidths=lw,
                       facecolors=dg_colors, edgecolors="0.7",
                       zorder=8)

            ax.errorbar(self.lssfr[where], self.hostmass[where],
                        xerr=[self.lssfr_err_d[where],
                              self.lssfr_err_u[where]],
                        yerr=self.hostmass_err[where],
                        ecolor='0.7', alpha=ealpha, ms=0,
                        lw=elw,
                        ls='none', label=None, zorder=5)

            ax.vline(lssfr_med,
                     color='0', alpha=1, linewidth=lw)

            ax.set_ylim([self.floor, self.ceil])

            ax.set_xlabel(r'$\mathrm{log(LsSFR)}$', fontsize=fontsize)
            ax.set_ylabel(r'$\mathrm{M}_\mathrm{host}$', fontsize=fontsize)

        if model:
            self.show_model(ax=ax, shift=lssfr_med, lw=mod_lw)

        if show_leg:
            ax.legend(ncol=1, loc='upper left')

        if yotext:
            ax.text(ax.get_xlim()[0]*0.99, ax.get_ylim()[-1]*0.99,
                    'Old',
                    ha='left', va='top',
                    fontsize='x-large',
                    color=plt.cm.viridis(0.97, 0.5))

            ax.text(ax.get_xlim()[-1]*1.01, ax.get_ylim()[-1]*0.99,
                    'Young',
                    ha='right', va='top',
                    fontsize='x-large',
                    color=plt.cm.viridis(0.05, 0.5))

        # ----------------------------------------------------------- #
        #                           Modeler                           #
        # ----------------------------------------------------------- #

    def show_model(self, ax=None, shift=0,
                   facealpha=0.1, edgealpha=.8,
                   o_factor=-3, y_factor=2, **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[7, 3.5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        x_lin = np.linspace(self.floor, self.ceil, 3000)

        ax.fill_betweenx(x_lin,
                         y_factor*self.plot_y(x_lin)
                         + shift,
                         shift,
                         facecolor=plt.cm.viridis(0.05, facealpha),
                         edgecolor=plt.cm.viridis(0.05, edgealpha),
                         label='model young', **kwargs)

        ax.fill_betweenx(x_lin,
                         o_factor*self.plot_o(x_lin)
                         + shift,
                         shift,
                         facecolor=plt.cm.viridis(0.95, facealpha),
                         edgecolor=plt.cm.viridis(0.95, edgealpha),
                         label='model old', **kwargs)

        # ----------------------------------------------------------- #
        #                           Histoer                           #
        # ----------------------------------------------------------- #

    def show_hist(self, ax=None, bottom=0,
                  range=(8, 12), bins=14, lw=1,
                  yotext=True):
        """Shows the weighted hist of the model"""
        if ax is None:
            fig = plt.figure(figsize=[7, 3.5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        prop = dict(orientation='horizontal',
                    histtype='step',
                    fill=True,
                    range=range, bins=bins,
                    lw=lw)

        self.amp = (prop['range'][1]-prop['range'][0])/prop['bins']

        ax.hist(self.hostmass, weights=(self.py),
                facecolor=plt.cm.viridis(0.05, 0.5),
                edgecolor='0.7', bottom=bottom,
                **prop)
        ax.hist(self.hostmass, weights=self.py-1,
                facecolor=plt.cm.viridis(0.95, 0.5),
                edgecolor='0.7', bottom=bottom,
                **prop)

        if yotext:
            ax.text(-1.30, ax.get_ylim()[-1]*0.99,
                    'Old',
                    ha='right', va='top',
                    fontsize='x-large',
                    color=plt.cm.viridis(0.97, 0.5))

            ax.text(1.30, ax.get_ylim()[-1]*0.99,
                    'Young',
                    ha='left', va='top',
                    fontsize='x-large',
                    color=plt.cm.viridis(0.05, 0.5))

        # ----------------------------------------------------------- #
        #                           Totaler                           #
        # ----------------------------------------------------------- #

    def show_model_tot(self, ax=None, fontsize='large', **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[7, 3.5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        x_lin = np.linspace(self.floor, self.ceil, 3000)

        ax.plot(x_lin,
                self.plot_a(x_lin),
                color="C2",
                label='model', **kwargs)

        ax.hist(self.pd.hostmass, density=True, histtype='step',
                color="0.5", alpha=1, zorder=8)
        ax.vline(self.param['mu'],
                 ymin=0, ymax=np.max(self.plot_a(x_lin)),
                 color="C2")

        ax.set_xlabel(r'$\mathrm{M}_\mathrm{host}$', fontsize=fontsize)
        ax.set_ylabel(r'$\mathrm{Probability}$', fontsize=fontsize)


# =========================================================================== #
#                              Dblegauss + gauss                              #
# =========================================================================== #


class Evol3G3M3S(Evol3G3M4S):
    '''3G3M3S'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_2', 'sigma_2', 'mu_3', 'sigma_3']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    PLTALL = ["self.param['" + k + "']" for k in FREEPARAMETERS]
    GUESSVAL = [9.5, 2, 0.5, 9, 0.5, 10.5, 1, 0.5]
    GUESS = [k + '=' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    def likelihood_o(self, x, dx, a, mu_2, sigma_2, mu_3, sigma_3):
        '''Underlying mass distribution of old SNe Ia'''
        return a*self.gauss(x, dx, mu_2, sigma_2) + \
            (1-a)*self.gauss(x, dx, mu_3, sigma_3)

    def likelihood_tot(self, info, x, dx, mu_1, sigma_1,
                       a, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, a, mu_2, sigma_2, mu_3, sigma_3)

    def loglikelihood(self, mu_1, sigma_1,
                      a, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.hostmass,
                                                    self.hostmass_err,
                                                    mu_1, sigma_1,
                                                    a, mu_2, sigma_2,
                                                    mu_3, sigma_3)))

# =========================================================================== #
#                                Gauss + gauss                                #
# =========================================================================== #


class Evol2G2M2S(Evol3G3M4S):
    '''2G2M2S'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]
    PLTALL = ["self.param['" + k + "']" for k in FREEPARAMETERS]
    GUESSVAL = [9.5, 1, 10.5, 1]
    GUESS = [k + '=' + str(v) for k, v in zip(FREEPARAMETERS, GUESSVAL)]
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    def likelihood_o(self, x, dx, mu_2, sigma_2):
        '''Underlying mass distribution of old SNe Ia'''
        return self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_tot(self, info, x, dx, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, mu_2, sigma_2)

    def loglikelihood(self, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.hostmass,
                                                    self.hostmass_err,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2)))
