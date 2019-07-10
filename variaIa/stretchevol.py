#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import iminuit as im
import matplotlib.pyplot as plt
plt.style.use(['classic', 'seaborn-white'])

lssfr_med = -10.82


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

#   ###########################################################################
#   ################################ STRETCHDIST ##############################
#   ###########################################################################


class StretchDist():
    ''' '''

    def set_data(self, redshifts, stretchs, stretchs_err):
        '''Donne les données des redshifts, stretchs, et stretchs_err'''
        self.redshifts = redshifts
        self.stretchs = stretchs
        self.stretchs_err = stretchs_err

        self.floor = np.floor(np.min(self.stretchs)-0.4)
        self.ceil = np.ceil(np.max(self.stretchs)+0.3)

#   ###########################################################################
#   ################################# LSSFRDIST ###############################
#   ###########################################################################


class LssfrStretchDist(StretchDist):
    ''' '''

    def set_lssfr(self, stretch, stretch_err,
                  lssfr, lssfr_err_d, lssfr_err_u, py):
        '''Donne les données de lssfr et erreurs up/down + proba j/v'''
        self.stretch = stretch
        self.stretch_err = stretch_err

        self.lssfr = lssfr
        self.lssfr_err_d = lssfr_err_d
        self.lssfr_err_u = lssfr_err_u

        self.py = py

        self.floor = np.floor(np.min(self.stretch)-0.4)
        self.ceil = np.ceil(np.max(self.stretch)+0.3)

#   ###########################################################################
#   ################################ EVOLHOWELL ###############################
#   ###########################################################################


class Evol2G2M2S(LssfrStretchDist):
    '''USAGE :
    evol = stretchevol.Evol2G2M2S()

    evol.set_data(redshifts, stretchs, stretchs_err)
    evol.set_lssfr(lssfr, lssfr_err_d, lssfr_err_u, py), optional

    evol.minimize()

    evol.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

    def __new__(cls, *arg, **kwargs):
        '''Upgrade of the New function to enable the _minuit_ black magic'''
        obj = super(Evol2G2M2S, cls).__new__(cls)

        exec("@make_method(Evol2G2M2S)\n" +
             "def logprob(self, %s):\n" % (", ".join(obj.FREEPARAMETERS)) +
             "    loglikelihood = self.loglikelihood(%s)\n" %
             (", ".join(obj.FREEPARAMETERS)) +
             "    logprior = self.logprior()\n" +
             "    return logprior + loglikelihood")

        if len(obj.YOUNGPARAMETERS) == 0:
            return obj

        exec("@make_method(Evol2G2M2S)\n" +
             "def plotter(self, name=None):\n" +
             "   '''Trace les fits et les données si lssfr donné'''\n" +
             "   dgmap = plt.cm.get_cmap('viridis')\n" +
             "\n" +
             "   if hasattr(self, 'lssfr'):\n" +
             "       dg_colors = [dgmap(i) for i in (1-self.py)]\n" +
             "       plt.scatter(self.lssfr, self.stretch, marker='o',\n" +
             "                   s=100, linewidths=0.5,\n" +
             "                   facecolors=dg_colors, edgecolors='0.7',\n" +
             "                   label='SNe data', zorder=8)\n" +
             "\n" +
             "       plt.errorbar(self.lssfr, self.stretch,\n" +
             "                    xerr=[self.lssfr_err_d, self.lssfr_err_u],\n"
             +
             "                    yerr=self.stretch_err,\n" +
             "                    ecolor='0.7', alpha=1, ms=0,\n" +
             "                    ls='none', label=None, zorder=5)\n" +
             "\n" +
             "   plt.axvline(lssfr_med,\n" +
             "               color='0.7', alpha=.5, linewidth=2.0)\n" +
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
             "   ax = plt.gca()\n" +
             "\n" +
             "   ax.tick_params(axis='both',\n" +
             "                  direction='in',\n" +
             "                  length=10, width=3,\n" +
             "                  labelsize=20,\n" +
             "                  which='both',\n" +
             "                  top=True, right=True)\n" +
             "\n" +
             "   ax.set_ylim([self.floor, self.ceil])\n" +
             "\n" +
             "   plt.xlabel('$LsSFR$', fontsize=20)\n" +
             "   plt.ylabel('$x_1$', fontsize=20)\n" +
             "\n" +
             "   plt.legend(ncol=1, loc='upper left')\n" +
             "\n" +
             "   plt.title(name, fontsize=20)\n" +
             "\n" +
             "   plt.show()")

        return obj

#   ################################# SETTER ##################################

    def set_param(self, param):
        self.param = {k: v for k, v in zip(self.FREEPARAMETERS,
                                           np.atleast_1d(param))}

#   ################################# EXTMOD ##################################

    def delta(self, z):
        '''Gives the fraction of young SNe Ia as a function of redshift;
        taken from https://arxiv.org/abs/1806.03849'''
        K = 0.87
        Phi = 2.8
        return (K**(-1)*(1+z)**(-Phi)+1)**(-1)

#   ################################## FITTER #################################

    def gauss(self, x, dx, mu, sigma):
        '''Le modèle de distribution'''
        return scipy.stats.norm.pdf(x, mu, scale=np.sqrt(dx**2+sigma**2))

    def likelihood_y(self, x, dx, mu_1, sigma_1):
        '''La fonction décrivant le modèle des SNe jeunes'''
        return self.gauss(x, dx, mu_1, sigma_1)

    def likelihood_o(self, x, dx, mu_2, sigma_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_tot(self, z, x, dx, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return self.delta(z)*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-self.delta(z))*self.likelihood_o(x, dx, mu_2, sigma_2)

    def loglikelihood(self, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.redshifts,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2)))

    def logprior(self):
        '''Impose des contraintes sur les paramètres'''
        return 0

    def minimize(self, print_level=0, **kwargs):
        '''Renvoie la meilleure valeur des paramètres'''
        self.m_tot = im.Minuit(self.logprob,
                               print_level=print_level,
                               pedantic=False,
                               **kwargs)

        self.migrad_out = self.m_tot.migrad()

        self.set_param([self.m_tot.values[k] for k in self.FREEPARAMETERS])

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

#   ################################# PLOTTER #################################
    """
    def scatter(self, model=True):
        '''Trace le nuage de points et les fits
        model=False ne montre que les données du pandas'''
        dgmap = plt.cm.get_cmap('viridis')
        dg_colors = [dgmap(i) for i in (1-self.py)]

        if hasattr(self, 'lssfr'):
            plt.scatter(self.lssfr, self.stretch, marker='o',
                        s=100, linewidths=0.5,
                        facecolors=dg_colors, edgecolors="0.7",
                        label='SNe data', zorder=8)

            plt.errorbar(self.lssfr, self.stretch,
                         xerr=[self.lssfr_err_d, self.lssfr_err_u],
                         yerr=self.stretch_err,
                         ecolor='0.7', alpha=1, ms=0,
                         ls='none', label=None, zorder=5)

        plt.axvline(lssfr_med,
                    color='0.7', alpha=.5, linewidth=2.0)

        x_linspace = np.linspace(self.floor, self.ceil, 3000)

        if model is True:
            plt.fill_betweenx(x_linspace,
                              2*self.likelihood_y(x_linspace, 0,
                                                  self.param['mu_1'],
                                                  self.param['sigma_1'])
                              + lssfr_med,
                              lssfr_med,
                              facecolor=plt.cm.viridis(0.05, 0.1),
                              edgecolor=plt.cm.viridis(0.05, 0.8),
                              lw=2, label='model young')

            plt.fill_betweenx(x_linspace,
                              -3*self.likelihood_o(x_linspace, 0,
                                                   self.param['mu_2'],
                                                   self.param['sigma_2'])
                              + lssfr_med,
                              lssfr_med,
                              facecolor=plt.cm.viridis(0.95, 0.1),
                              edgecolor=plt.cm.viridis(0.95, 0.8),
                              lw=2, label='model old')

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

        plt.legend(ncol=1, loc='upper left')

        plt.title('1GSNF model', fontsize=20)

        plt.show()
    """

#   ###########################################################################
#   ################################## EVOLHOWF ###############################
#   ###########################################################################


class Evol2G2M2SF(Evol2G2M2S):
    '''USAGE :
    evol = stretchevol.Evol2G2M2SF()

    evol.set_data(redshifts, stretchs, stretchs_err)
    evol.set_lssfr(lssfr, lssfr_err_d, lssfr_err_u, py), optional

    evol.minimize()

    evol.scatter()
    '''

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

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

#   ###########################################################################
#   ################################ EVOLSIMPLE ###############################
#   ###########################################################################


class Evol1G1M1S(Evol2G2M2S):
    '''USAGE :
    evolS = stretchevol.Evol1G1M1S()
    evolS.set_data(redshifts, stretchs, stretchs_err)

    evolS.minimize()

    evolS.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigma']

#   ################################## FITTER #################################

    def likelihood_tot(self, x, dx, mu, sigma):
        '''La fonction de distribution'''
        return self.gauss(x, dx, mu, sigma)

    def loglikelihood(self, mu, sigma):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    mu, sigma)))


#   ###########################################################################
#   ################################ EVOLKESSLE ###############################
#   ###########################################################################


class Evol2G1M2S(Evol2G2M2S):
    '''USAGE :
    evolK = stretchevol.Evol2G1M2S()
    evolK.set_data(redshifts, stretchs, stretchs_err)

    evolK.minimize()

    evolK.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigma_m', 'sigma_p']

#   ################################## FITTER #################################

    def likelihood_tot(self, x, dx, mu, sigma_m, sigma_p):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        flag_up = x >= mu
        likelihood = np.zeros(len(x))

        likelihood[flag_up] = self.gauss(x[flag_up], dx[flag_up],
                                         mu, sigma_p)
        likelihood[~flag_up] = self.gauss(x[~flag_up], dx[~flag_up],
                                          mu, sigma_m)
        return likelihood

    def loglikelihood(self, mu, sigma_m, sigma_p):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.stretchs,
                                                    self.stretchs_err,
                                                    mu,
                                                    sigma_m, sigma_p)))

#   ###########################################################################
#   ################################# EVOLNR1S ################################
#   ###########################################################################


class Evol3G2M1S(Evol2G2M2S):
    '''USAGE :
    evol = stretchevol.Evol2G2M1S()

    evol.set_data(redshifts, stretchs, stretchs_err)
    evol.set_lssfr(lssfr, lssfr_err_d, lssfr_err_u, py), optional

    evol.minimize()

    evol.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

    def likelihood_o(self, x, dx, a, mu_1, sigma_1, mu_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_1, sigma_1) + \
            (1-a)*self.gauss(x, dx, mu_2, sigma_1)

    def likelihood_tot(self, z, x, dx, a, mu_1, sigma_1, mu_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return self.delta(z)*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-self.delta(z))*self.likelihood_o(x, dx, a, mu_1, sigma_1, mu_2)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.redshifts,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2)))

#   ###########################################################################
#   ############################### EVOLNR1SSNF ###############################
#   ###########################################################################


class Evol3G2M1SSNF(Evol3G2M1S):
    '''USAGE :
    evol = stretchevol.Evol3G2M1SSNF()

    evol.set_lssfr(stretch, stretch_err,
                   lssfr, lssfr_err_d, lssfr_err_u, py)

    evol.minimize()

    evol.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

    def likelihood_tot(self, py, x, dx, a, mu_1, sigma_1, mu_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return py*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-py)*self.likelihood_o(x, dx, a, mu_1, sigma_1, mu_2)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.py,
                                                    self.stretch,
                                                    self.stretch_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2)))


#   ###########################################################################
#   ################################# EVOLNR1SF ###############################
#   ###########################################################################


class Evol3G2M1SF(Evol3G2M1S):
    '''USAGE :
    evol = stretchevol.Evol2G1SF()

    evol.set_data(redshifts, stretchs, stretchs_err)
    evol.set_lssfr(lssfr, lssfr_err_d, lssfr_err_u, py), optional

    evol.minimize()

    evol.scatter()
    '''

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2']
    FREEPARAMETERS = np.append(GLOBALPARAMETERS, OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

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

#   ###########################################################################
#   ################################# EVOLNR2S ################################
#   ###########################################################################


class Evol3G2M2S(Evol2G2M2S):
    '''USAGE :
    evol = stretchevol.Evol3G2M2S()

    evol.set_data(redshifts, stretchs, stretchs_err)
    evol.set_lssfr(lssfr, lssfr_err_d, lssfr_err_u, py), optional

    evol.minimize()

    evol.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    FREEPARAMETERS = OLDPARAMETERS
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

    def likelihood_o(self, x, dx, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_1, sigma_1) + \
            (1-a)*self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_tot(self, z, x, dx, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return self.delta(z)*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-self.delta(z))*self.likelihood_o(x, dx, a, mu_1, sigma_1,
                                                mu_2, sigma_2)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2, sigma_2):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.redshifts,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2, sigma_2)))

#   ###########################################################################
#   ################################ EVOLNR2SF ################################
#   ###########################################################################


class Evol3G2M2SF(Evol3G2M2S):
    '''USAGE :
    evolF = stretchevol.Evol3G2M2SF()
    evolF.set_data(redshifts, stretchs, stretchs_err)

    evolF.minimize()

    evolF.scatter()
    '''

    GLOBALPARAMETERS = ['f']
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(GLOBALPARAMETERS, OLDPARAMETERS)
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

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

#   ###########################################################################
#   ################################ EVOLNRTOT ################################
#   ###########################################################################


class Evol3G3M3S(Evol3G2M2S):
    '''USAGE :
    evolF = stretchevol.Evol3G3M3S()
    evolF.set_data(redshifts, stretchs, stretchs_err)

    evolF.minimize()

    evolF.scatter()
    '''

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['a', 'mu_2', 'sigma_2', 'mu_3', 'sigma_3']
    FREEPARAMETERS = ['a',
                      'mu_1', 'sigma_1',
                      'mu_2', 'sigma_2',
                      'mu_3', 'sigma_3']
    PLTYOUNG = ["self.param['" + k + "']" for k in YOUNGPARAMETERS]
    PLTOLD = ["self.param['" + k + "']" for k in OLDPARAMETERS]

#   ################################## FITTER #################################

    def likelihood_o(self, x, dx, a, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction décrivant le modèle des SNe vieilles'''
        return a*self.gauss(x, dx, mu_2, sigma_2) + \
            (1-a)*self.gauss(x, dx, mu_3, sigma_3)

    def likelihood_tot(self, z, x, dx, a,
                       mu_1, sigma_1,
                       mu_2, sigma_2,
                       mu_3, sigma_3):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return self.delta(z)*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-self.delta(z))*self.likelihood_o(x, dx, a,
                                                mu_2, sigma_2,
                                                mu_3, sigma_3)

    def loglikelihood(self, a, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.redshifts,
                                                    self.stretchs,
                                                    self.stretchs_err,
                                                    a,
                                                    mu_1, sigma_1,
                                                    mu_2, sigma_2,
                                                    mu_3, sigma_3)))

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
