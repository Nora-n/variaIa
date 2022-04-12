#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import scipy
import iminuit
import numpy as np
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
#                                  GENERIQUE                                  #
#                                                                             #
# =========================================================================== #


class generic():
    '''Usage:
       gen = stretchevol.generic()
       gen.set_model('model')
       fitted_model = gen.fit(pandas)

       fitted_model.param'''

    def set_model(self, classname):
        '''Associates an uninstantiated class to
        `self.model` from its string'''
        self.model = getattr(sys.modules[__name__], classname)

    def fit(self, pandas, **kwargs):
        '''Instantiates the class with the pandas,
        apply the `minimize` method,
        and gives that back as an output'''
        model = self.model(pandas)
        model.minimize(**kwargs)
        return(model)


# =========================================================================== #
#                                                                             #
#                                    MODELS                                   #
#                                                                             #
# =========================================================================== #

# =========================================================================== #
#                                 EvolHowell                                  #
# =========================================================================== #


class Evol2G2M2S():
    '''Howell+drift'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    GUESSVAL = [snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = False

    # =================================================================== #
    #                               Initial                               #
    # =================================================================== #

    def __init__(self, pandas, py=True):
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

        self.floor = np.floor(np.min(self.stretchs)-0.3)
        self.ceil = np.ceil(np.max(self.stretchs)+0.3)

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
        self.pltyoung = {k: self.param[k] for k in self.YOUNGPARAMETERS}
        self.pltold = {k: self.param[k] for k in self.OLDPARAMETERS}
        self.pltall = {k: self.param[k] for k in self.FREEPARAMETERS}

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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

    def logprob(self):
        return self.logprior() + self.loglikelihood(*self.FREEPARAMETERS)

    def get_logl(self, **kwargs):
        if not hasattr(self, 'migrad_out'):
            self.minimize(**kwargs)
        return self.migrad_out[0]['fval']

    def get_bic(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return mdlogl + k*np.log(len(self.stretchs))

    def get_aic(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return 2*k + mdlogl

    # ------------------------------------------------------------------- #
    #                              MINIMIZER                              #
    # ------------------------------------------------------------------- #

    def minimize(self, guess=None, limits=None, **kwargs):
        '''Renvoie la meilleure valeur des paramètres'''
        if guess is None:
            self.m_tot = iminuit.Minuit(self.loglikelihood,
                                        **self.GUESS,
                                        **kwargs)
        else:
            self.m_tot = iminuit.Minuit(self.loglikelihood,
                                        guess,
                                        **kwargs)

        if limits is None:
            pass
        else:
            self.m_tot.limits = limits

        self.m_tot.errordef = iminuit.Minuit.LIKELIHOOD
        self.migrad_out = self.m_tot.migrad()

        self.set_param([self.m_tot.values[k] for k in self.FREEPARAMETERS])

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

        # ----------------------------------------------------------- #
        #                           Generic                           #
        # ----------------------------------------------------------- #

    def plot_a(self, x_lin):
        return self.likelihood_tot(x_lin,
                                   np.zeros(len(x_lin)),
                                   **self.pltall)

    def plot_y(self, x_lin):
        return self.likelihood_y(x_lin,
                                 np.zeros(len(x_lin)),
                                 **self.pltyoung)

    def plot_o(self, x_lin):
        return self.likelihood_o(x_lin,
                                 np.zeros(len(x_lin)),
                                 **self.pltold)

        # ----------------------------------------------------------- #
        #                           Scatter                           #
        # ----------------------------------------------------------- #

    def scatter(self, model=True, ax=None, show_leg=True, lssfr=True,
                mod_lw=2, lw=.5, elw=1, ealpha=1, s=80,
                facealpha=1, fontsize='large'):
        '''Trace le nuage de points et les fits
        model=False ne montre que les données du pandas'''
        dgmap = plt.cm.get_cmap('viridis')

        if ax is None:
            fig = plt.figure(figsize=[8, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        if lssfr:
            where = self.pd['lssfr'] != 0
            dg_colors = [dgmap(i, facealpha) for i in (1-self.py[where])]
            ax.scatter(self.lssfr[where], self.stretchs[where],
                       marker='o',
                       s=s, linewidths=lw,
                       facecolors=dg_colors, edgecolors="0.7",
                       zorder=8)

            ax.errorbar(self.lssfr[where], self.stretchs[where],
                        xerr=[self.lssfr_err_d[where],
                              self.lssfr_err_u[where]],
                        yerr=self.stretchs_err[where],
                        ecolor='0.7', alpha=ealpha, ms=0,
                        lw=elw,
                        ls='none', label=None, zorder=5)

            ax.vline(lssfr_med,
                     color='0', alpha=1, linewidth=lw)

            ax.set_ylim([self.floor, self.ceil])

            ax.set_xlabel(r'$\mathrm{log(LsSFR)}$', fontsize=fontsize)
            ax.set_ylabel(r'$\mathrm{x}_1$', fontsize=fontsize)

        if model:
            self.show_model(ax=ax, shift=lssfr_med, lw=mod_lw)

        if show_leg:
            ax.legend(ncol=1, loc='upper left')

        # plt.title('1GSNF model', fontsize=20)

        # ----------------------------------------------------------- #
        #                           Modeler                           #
        # ----------------------------------------------------------- #

    def show_model(self, ax=None, shift=0,
                   facealpha=0.1, edgealpha=.8,
                   o_factor=-3, y_factor=2, **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[8, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        x_linspace = np.linspace(self.floor, self.ceil, 3000)

        ax.fill_betweenx(x_linspace,
                         y_factor*self.plot_y(x_linspace)
                         + shift,
                         shift,
                         facecolor=plt.cm.viridis(0.05, facealpha),
                         edgecolor=plt.cm.viridis(0.05, edgealpha),
                         label='model young', **kwargs)

        ax.fill_betweenx(x_linspace,
                         o_factor*self.plot_o(x_linspace)
                         + shift,
                         shift,
                         facecolor=plt.cm.viridis(0.98, facealpha),
                         edgecolor=plt.cm.viridis(0.98, edgealpha),
                         label='model old', **kwargs)

        # ----------------------------------------------------------- #
        #                           Totaler                           #
        # ----------------------------------------------------------- #

    def show_model_tot(self, ax=None, fontsize='large', **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[8, 5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        x_linspace = np.linspace(self.floor, self.ceil, 3000)

        ax.plot(x_linspace,
                self.plot_a(x_linspace),
                color="C2",
                label='model', **kwargs)

        ax.hist(self.pd.stretchs, density=True, histtype='step',
                color="0.5", alpha=1, zorder=8)
        ax.vline(self.param['mu'],
                 ymin=0, ymax=np.max(self.plot_a(x_linspace)),
                 color="C2")

        ax.set_xlabel(r'$\mathrm{x}_1$', fontsize=fontsize)
        ax.set_ylabel(r'$\mathrm{Probability}$', fontsize=fontsize)

# =========================================================================== #
#                                 EvolHowellF                                 #
# =========================================================================== #


class Evol2G2M2SF(Evol2G2M2S):
    '''Howell+const'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    GUESSVAL = [snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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
    '''Asymmetric'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigma_m', 'sigma_p']
    GUESSVAL = [0.973, 1.5, 0.5]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

    def loglikelihood(self, a, mu_1, sigma_1, mu_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
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
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    @staticmethod
    def get_a(aa):
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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
    '''Base+const'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(GLOBALPARAMETERS, OLDPARAMETERS)
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

    def loglikelihood(self, f, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    f, a,
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
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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
    GUESSVAL = [snf_a, snf_mu_1, snf_sigma_1, snf_mu_2, snf_sigma_2]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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
    @staticmethod
    def delta(z):
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
