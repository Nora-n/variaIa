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
        '''Associates an uninstantiated class to
        `self.model` from its string'''
        self.model = getattr(sys.modules[__name__], classname)

    def fit(self, pandas, guess=None, limits=None, **kwargs):
        '''Instantiates the class with the pandas,
        apply the `minimize` method,
        and gives that back as an output'''
        model = self.model(pandas)
        model.minimize(guess, limits, **kwargs)
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
    GUESSVAL = [9.5, 2, 0.5, 9, 0.5, 10.5, 1, 0.5]
    LIMVAL = [None, None, (0, 1),
              None, None, None, None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = False

    # =================================================================== #
    #                               Initial                               #
    # =================================================================== #

    def __init__(self, pandas, py=True):
        '''Pour une meilleure utilisation des données'''
        self.pd = pandas.loc[pandas['hostmass'] > 7]

        self.redshifts = pandas.redshifts
        self.hostmass = pandas.hostmass
        # self.hostmass_err = np.sqrt(pandas.hostmass_err**2+0.1**2)
        self.hostmass_err = pandas.hostmass_err

        if py:
            self.info = pandas.infor.values
        else:
            self.info = self.delta(pandas.redshifts)

        self.py = pandas.py
        self.lssfr = pandas.lssfr
        self.lssfr_err_d = pandas.lssfr_err_d
        self.lssfr_err_u = pandas.lssfr_err_u

        self.floor = np.floor(np.min(self.hostmass))
        self.ceil = np.ceil(np.max(self.hostmass))

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
        self.pltmy = {k: self.param[k]
                      for k in self.YOUNGPARAMETERS
                      if 'mu' in k}
        self.pltold = {k: self.param[k] for k in self.OLDPARAMETERS}
        self.pltmo = {k: self.param[k]
                      for k in self.OLDPARAMETERS
                      if 'mu' in k}
        self.pltall = {k: self.param[k] for k in self.FREEPARAMETERS}
        self.pltma = {k: self.param[k]
                      for k in self.FREEPARAMETERS
                      if 'mu' in k}

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

    # ------------------------------------------------------------------- #
    #                              PROBALIZER                             #
    # ------------------------------------------------------------------- #

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

    def logprob(self):
        return self.logprior() + self.loglikelihood(*self.FREEPARAMETERS)

    def get_logl(self, **kwargs):
        if not hasattr(self, 'migrad_out'):
            self.minimize(**kwargs)
        return self.migrad_out.fval

    def get_bic(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return mdlogl + k*np.log(len(self.hostmass))

    def get_aic(self):
        k = len(self.FREEPARAMETERS)
        mdlogl = self.get_logl()

        return 2*k + mdlogl

    # ------------------------------------------------------------------- #
    #                              MINIMIZER                              #
    # ------------------------------------------------------------------- #

    def minimize(self, guess, limits, **kwargs):
        '''Renvoie la meilleure valeur des paramètres'''
        if guess is None:
            self.m_tot = iminuit.Minuit(self.loglikelihood,
                                        **self.GUESS,
                                        **kwargs)
        else:
            self.m_tot = iminuit.Minuit(self.loglikelihood,
                                        guess,
                                        **kwargs)

        if limits == 'auto':
            self.m_tot.limits = self.LIMVAL
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

    def plot_a(self, M_lin, z=None, f=None):
        if self.FIXED:
            return(self.likelihood_tot(M_lin,
                                       np.zeros(len(M_lin)),
                                       **self.pltall))
        else:
            if (z is None) and (f is not None):
                return self.likelihood_tot(f,
                                           M_lin,
                                           np.zeros(len(M_lin)),
                                           **self.pltall)
            elif (z is not None) and (f is None):
                return self.likelihood_tot(self.delta(z),
                                           M_lin,
                                           np.zeros(len(M_lin)),
                                           **self.pltall)
            elif (z is None) and (f is None):
                raise NameError('Either `z` or `f` must be given')
            elif (z is not None) and (f is not None):
                raise NameError("`z` and `f` can't both be given")

    def plot_ga(self, Mlin,
                z=None, f=None,
                g_ls_y='--', g_ls_o='-.',
                g_c_y=plt.cm.viridis(0.05),
                g_c_o=plt.cm.viridis(0.98)):
        if (z is None) and (f is not None):
            if bool(self.plot_gy(Mlin)):
                mode1 = f*np.asarray(self.plot_gy(Mlin))
            else:
                mode1 = f*np.asarray([self.plot_y(Mlin)])
            ls_y = [g_ls_y for i in range(len(mode1))]
            c_y = [g_c_y for i in range(len(mode1))]
            if bool(self.plot_go(Mlin)):
                mode2 = (1-f)*np.asarray(self.plot_go(Mlin))
            else:
                mode2 = (1-f)*np.asarray([self.plot_o(Mlin)])
            ls_o = [g_ls_o for i in range(len(mode2))]
            c_o = [g_c_o for i in range(len(mode2))]
            return (np.vstack((mode1, mode2)),
                    np.concatenate((ls_y, ls_o)),
                    np.concatenate((c_y, c_o)))
        elif (z is not None) and (f is None):
            if bool(self.plot_gy(Mlin)):
                mode1 = self.delta(z)*np.asarray(self.plot_gy(Mlin))
            else:
                mode1 = self.delta(z)*np.asarray([self.plot_y(Mlin)])
            ls_y = [g_ls_y for i in range(len(mode1))]
            c_y = [g_c_y for i in range(len(mode1))]
            if bool(self.plot_go(Mlin)):
                mode2 = (1-self.delta(z))*np.asarray(self.plot_go(Mlin))
            else:
                mode2 = (1-self.delta(z))*np.asarray([self.plot_o(Mlin)])
            ls_o = [g_ls_o for i in range(len(mode2))]
            c_o = [g_c_o for i in range(len(mode2))]
            return (np.vstack((mode1, mode2)),
                    np.concatenate((ls_y, ls_o)),
                    np.concatenate((c_y, c_o)))
        elif (z is None) and (f is None):
            raise NameError('Either `z` or `f` must be given')
        elif (z is not None) and (f is not None):
            raise NameError("`z` and `f` can't both be given")

    def plot_evol(self, z=None, f=None):
        pnums = [p.split('_')[-1] for p in self.YOUNGPARAMETERS]
        pnums = np.unique([p for p in pnums if len(p) == 1])
        if len(pnums) == 2:
            bb = [self.param[k] for k in self.YOUNGPARAMETERS if '_' not in k]
            b = bb[0]
            mode1 = b*self.param['mu_'+pnums[0]] +\
                (1-b)*self.param['mu_'+pnums[1]]
        elif len(pnums) == 1:
            mode1 = self.param['mu_'+pnums[0]]
        elif len(pnums) == 0:
            return self.pltma['mu']
        pnums = [p.split('_')[-1] for p in self.OLDPARAMETERS]
        pnums = np.unique([p for p in pnums if len(p) == 1])
        if len(pnums) == 2:
            aa = [self.param[k] for k in self.OLDPARAMETERS if '_' not in k]
            a = aa[0]
            mode2 = a*self.param['mu_'+pnums[0]] +\
                (1-a)*self.param['mu_'+pnums[1]]
        else:
            mode2 = self.param['mu_'+pnums[0]]
        if self.FIXED:
            f = self.param['f']
            return f*mode1 + (1-f)*mode2
        else:
            if (z is None) and (f is not None):
                return f*mode1 + (1-f)*mode2
            elif (z is not None) and (f is None):
                return self.delta(z)*mode1 + (1-self.delta(z))*mode2
            elif (z is None) and (f is None):
                raise NameError('Either `z` or `f` must be given')
            elif (z is not None) and (f is not None):
                raise NameError("`z` and `f` can't both be given")

    def plot_y(self, M_lin):
        return self.likelihood_y(M_lin,
                                 np.zeros(len(M_lin)),
                                 **self.pltyoung)

    def plot_gy(self, Mlin):
        aa = [self.param[k] for k in self.YOUNGPARAMETERS if '_' not in k]
        if bool(aa):
            a = aa[0]
            pnums = [p.split('_')[-1] for p in self.YOUNGPARAMETERS]
            pnums = np.unique([p for p in pnums if len(p) == 1])
            return [a*self.gauss(Mlin, 0,
                                 self.pltyoung['mu_'+pnums[0]],
                                 self.pltyoung['sigma_'+pnums[0]]),
                    (1-a)*self.gauss(Mlin, 0,
                                     self.pltyoung['mu_'+pnums[1]],
                                     self.pltyoung['sigma_'+pnums[1]])]
        else:
            return []

    def plot_o(self, M_lin):
        return self.likelihood_o(M_lin,
                                 np.zeros(len(M_lin)),
                                 **self.pltold)

    def plot_go(self, Mlin):
        aa = [self.param[k] for k in self.OLDPARAMETERS if '_' not in k]
        if bool(aa):
            a = aa[0]
            pnums = [p.split('_')[-1] for p in self.OLDPARAMETERS]
            pnums = np.unique([p for p in pnums if len(p) == 1])
            return [a*self.gauss(Mlin, 0,
                                 self.pltold['mu_'+pnums[0]],
                                 self.pltold['sigma_'+pnums[0]]),
                    (1-a)*self.gauss(Mlin, 0,
                                     self.pltold['mu_'+pnums[1]],
                                     self.pltold['sigma_'+pnums[1]])]
        else:
            return []

        # ----------------------------------------------------------- #
        #                           Scatter                           #
        # ----------------------------------------------------------- #

    def scatter(self, ax=None, axh=None, bins=14,
                model=True, show_text=True,
                txt_old='Vieux', txt_yng='Jeune',
                show_leg=True, lssfr=True, shift=lssfr_med,
                mod_lw=1, lw=1, elw=1, ealpha=1, s=80,
                facealpha=1, fontsize='x-large'):
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

            if show_text:
                ax.text(ax.get_xlim()[0]*0.99, ax.get_ylim()[-1]*0.99,
                        txt_old,
                        ha='left', va='top',
                        fontsize='x-large',
                        color=plt.cm.viridis(1.00))

                ax.text(ax.get_xlim()[-1]*1.01, ax.get_ylim()[-1]*0.99,
                        txt_yng,
                        ha='right', va='top',
                        fontsize='x-large',
                        color=plt.cm.viridis(0.05, 0.5))

            ax.set_ylim([self.floor, self.ceil])

        if model:
            if axh is None:
                self.show_model(ax=ax, shift=lssfr_med,
                                lw=mod_lw, rotate=True,
                                legend=show_leg, zorder=8)
            else:
                prop = dict(orientation='horizontal',
                            histtype='step',
                            fill=True,
                            range=(7, 12), bins=bins,
                            lw=1)

                amp = (prop['range'][1] - prop['range'][0])/prop['bins']

                axh.hist(self.hostmass, weights=self.py,
                         facecolor=plt.cm.viridis(0.05, 0.5),
                         edgecolor='0.7',
                         **prop)
                axh.hist(self.hostmass, weights=(self.py-1),
                         facecolor=plt.cm.viridis(0.95, 0.5),
                         edgecolor="0.7",
                         **prop)

                self.show_model(ax=axh, rotate=True,
                                o_factor=-amp*np.sum(1-self.py),
                                y_factor=amp*np.sum(self.py),
                                facecolor_y='none', facecolor_o='none',
                                legend=False,
                                lw=1, zorder=8)
                axh.set_ylim(*ax.get_ylim())
                axh.set_xlabel('')
                axh.set_ylabel('')
                axh.set_yticks([])
                axh.set_xticks([])
                axh.set_frame_on(False)

                if show_text:
                    axh.text(-1.30, 12, txt_old,
                             ha='right', va='top',
                             fontsize='x-large',
                             color=plt.cm.viridis(1.00))

                    axh.text(1.30, 12, txt_yng,
                             ha='left', va='top',
                             fontsize='x-large',
                             color=plt.cm.viridis(0.05, 0.5))

            ax.set_xlabel(r'log(LsSFR)', fontsize=fontsize)
            ax.set_ylabel(r'$M_*$', fontsize=fontsize)

        if show_leg:
            ax.legend(ncol=1, loc='upper left')

        # ----------------------------------------------------------- #
        #                           Modeler                           #
        # ----------------------------------------------------------- #

    def show_model(self, ax=None, shift=0, rotate=False,
                   ls_y='-', ls_o='-',
                   facecolor_y=plt.cm.viridis(0.05, 0.1),
                   facecolor_o=plt.cm.viridis(0.98, 0.1),
                   edgecolor_y=plt.cm.viridis(0.05),
                   edgecolor_o=plt.cm.viridis(0.98),
                   o_label='Modèle vieux', y_label='Modèle jeune',
                   o_factor=-1, y_factor=1,
                   means=False, gauss=False,
                   legend=True, leg_kwargs={},
                   axes_l=True, y_l='Probabilité',
                   **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[6, 4])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        Mlin = np.linspace(self.floor, self.ceil, 3000)

        if rotate:
            ax.fill_betweenx(Mlin,
                             y_factor*self.plot_y(Mlin)
                             + shift,
                             shift,
                             ls=ls_y,
                             facecolor=facecolor_y,
                             edgecolor=edgecolor_y,
                             label=y_label, **kwargs)
            ax.fill_betweenx(Mlin,
                             o_factor*self.plot_o(Mlin)
                             + shift,
                             shift,
                             ls=ls_o,
                             facecolor=facecolor_o,
                             edgecolor=edgecolor_o,
                             label=o_label, **kwargs)
            ax.axvline(shift, color='k', lw=1, zorder=10)
            if axes_l:
                ax.set_ylabel('$M_*$', fontsize='x-large')
                ax.set_xlabel(y_l, fontsize='x-large')
            if means:
                self.show_means(ax=ax, shift=shift, rotate=rotate)
            else:
                pass

        else:
            ax.fill_between(Mlin,
                            y_factor*self.plot_y(Mlin)
                            + shift,
                            shift,
                            ls=ls_y,
                            facecolor=facecolor_y,
                            edgecolor=edgecolor_y,
                            label=y_label, **kwargs)
            ax.fill_between(Mlin,
                            o_factor*self.plot_o(Mlin)
                            + shift,
                            shift,
                            ls=ls_o,
                            facecolor=facecolor_o,
                            edgecolor=edgecolor_o,
                            label=o_label, **kwargs)
            ax.axhline(shift, color='k', lw=1, zorder=10)
            if axes_l:
                ax.set_ylabel(y_l, fontsize='x-large')
                ax.set_xlabel('$M_*$', fontsize='x-large')
            if means:
                self.show_means(ax=ax, shift=shift)
            if gauss:
                self.show_gauss(ax=ax, shift=shift)
        if legend:
            ax.legend(**leg_kwargs)
        return ax

        # ----------------------------------------------------------- #
        #                           Meansho                           #
        # ----------------------------------------------------------- #

    def show_means(self, ax=None,
                   z=None, f=None,
                   o_factor=-1, y_factor=1,
                   shift=0, rotate=False,
                   tot=False,
                   mu_c='k', mu_ls='--', mu_lw=1):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[6, 4])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        if (z is None) and (f is not None):
            y_fac = f
            o_fac = 1-f
        elif (z is not None) and (f is None):
            y_fac = self.delta(z)
            o_fac = 1-self.delta(z)
        elif (z is None) and (f is None):
            y_fac = y_factor
            o_fac = o_factor
        elif (z is not None) and (f is not None):
            raise NameError("`z` and `f` can't both be given")

        if not tot:
            if rotate:
                for muo in self.pltmo.values():
                    ax.hline(muo,
                             xmin=shift,
                             xmax=o_fac*(self.plot_o([muo])+shift),
                             color=mu_c, ls=mu_ls, lw=mu_lw, zorder=0)
                for muy in self.pltmy.values():
                    ax.hline(muy,
                             xmin=shift,
                             xmax=y_fac*(self.plot_y([muy])+shift),
                             color=mu_c, ls=mu_ls, lw=mu_lw, zorder=0)
            else:
                for muo in self.pltmo.values():
                    ax.vline(muo,
                             ymin=shift,
                             ymax=o_fac*(self.plot_o([muo])+shift),
                             color=mu_c, ls=mu_ls, lw=mu_lw, zorder=0)
                for muy in self.pltmy.values():
                    ax.vline(muy,
                             ymin=shift,
                             ymax=y_fac*(self.plot_y([muy])+shift),
                             color=mu_c, ls=mu_ls, lw=mu_lw, zorder=0)
        if tot:
            if (self.FIXED) and not (bool(self.GLOBALPARAMETERS)):
                Mlin = np.linspace(self.floor, self.ceil, 1000)
                ax.vline(self.param['mu'],
                         ymin=0, ymax=np.max(self.plot_a(Mlin)),
                         color=mu_c, ls=mu_ls, lw=mu_lw, zorder=0)
            else:
                for mua in self.pltma.values():
                    ax.vline(mua,
                             ymin=0, ymax=self.plot_a([mua], z, f),
                             color=mu_c, ls=mu_ls, lw=mu_lw, zorder=0)

        # ----------------------------------------------------------- #
        #                           Gaushow                           #
        # ----------------------------------------------------------- #

    def show_gauss(self, ax=None,
                   z=None, f=None,
                   o_factor=-1, y_factor=1,
                   shift=0,
                   g_c_o='0.7', g_ls_o=':',
                   g_c_y='0.7', g_ls_y='-.',
                   g_lw=1):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[6, 4])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])
        else:
            fig = ax.figure

        Mlin = np.linspace(self.floor, self.ceil, 3000)
        if (z is None) and (f is not None):
            y_fac = f
            o_fac = 1-f
        elif (z is not None) and (f is None):
            y_fac = self.delta(z)
            o_fac = 1-self.delta(z)
        elif (z is None) and (f is None):
            y_fac = y_factor
            o_fac = o_factor
        elif (z is not None) and (f is not None):
            raise NameError("`z` and `f` can't both be given")
        if bool(self.plot_gy(Mlin)):
            for toplot in self.plot_gy(Mlin):
                ax.plot(Mlin, y_fac*toplot+shift,
                        color=g_c_y, ls=g_ls_y, lw=g_lw)
        if bool(self.plot_go(Mlin)):
            for toplot in self.plot_go(Mlin):
                ax.plot(Mlin, o_fac*toplot+shift,
                        color=g_c_o, ls=g_ls_o, lw=g_lw)

        # ----------------------------------------------------------- #
        #                           Histoer                           #
        # ----------------------------------------------------------- #

    def show_hist(self, ax=None, bottom=0,
                  range=(8, 12), bins=14, lw=1,
                  fcy=plt.cm.viridis(0.05, 0.5),
                  fco=plt.cm.viridis(0.95, 0.5),
                  yotext=True, **kwargs):
        """Shows the weighted hist of the model"""
        if ax is None:
            fig = plt.figure(figsize=[7, 3.5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        prop = dict(orientation='horizontal',
                    histtype='step',
                    fill=True,
                    range=range, bins=bins,
                    lw=lw, **kwargs)

        self.amp = (prop['range'][1]-prop['range'][0])/prop['bins']

        ax.hist(self.hostmass, weights=(self.py),
                facecolor=fcy,
                edgecolor='0.7', bottom=bottom,
                **prop)
        ax.hist(self.hostmass, weights=self.py-1,
                facecolor=fco,
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

    def show_model_tot(self, ax=None, z=None, f=None,
                       color="C2", label='auto',
                       legend=True, leg_kwargs={},
                       show_hist=False, histtype='step', c_hist='0.5',
                       means=True, model=True, gauss=True,
                       fontsize='x-large', **kwargs):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[6, 4])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        else:
            fig = ax.figure

        Mlin = np.linspace(self.floor, self.ceil, 3000)

        if label is None:
            ax.plot(Mlin,
                    self.plot_a(Mlin, z, f),
                    color=color,
                    **kwargs)
        elif 'auto' in label:
            ax.plot(Mlin,
                    self.plot_a(Mlin, z, f),
                    color=color,
                    label=self.__doc__,
                    **kwargs)
        else:
            ax.plot(Mlin,
                    self.plot_a(Mlin, z, f),
                    color=color,
                    label=label,
                    **kwargs)
        if show_hist:
            ax.hist(self.pd.stretchs, density=True, histtype=histtype,
                    color=c_hist, alpha=1, zorder=8)
        if model:
            if (self.FIXED) and not (bool(self.GLOBALPARAMETERS)):
                if means:
                    self.show_means(ax=ax, z=z, f=f, tot=True)
                else:
                    pass
            if bool(self.GLOBALPARAMETERS):
                z = None
                f = self.param['f']
                self.show_model(ax=ax, y_factor=f, o_factor=1-f,
                                legend=legend)
            elif not self.FIXED:
                if (z is None) and (f is not None):
                    self.show_model(ax=ax, y_factor=f, o_factor=1-f,
                                    legend=legend)
                elif (z is not None) and (f is None):
                    self.show_model(ax=ax,
                                    y_factor=self.delta(z),
                                    o_factor=1-self.delta(z),
                                    legend=legend)
                elif (z is None) and (f is None):
                    raise NameError('Either `z` or `f` must be given')
                elif (z is not None) and (f is not None):
                    raise NameError("`z` and `f` can't both be given")
            if means:
                self.show_means(ax=ax, z=z, f=f)
            if gauss:
                self.show_gauss(ax=ax, z=z, f=f)
        else:
            if means:
                self.show_means(ax=ax, z=z, f=f, tot=True)

        ax.set_xlabel('$M_*$', fontsize=fontsize)
        ax.set_ylabel('Probabilité', fontsize=fontsize)

        if legend:
            ax.legend(**leg_kwargs)

        ax.set_ylim(bottom=0)

        return ax

        # ----------------------------------------------------------- #
        #                           Modevol                           #
        # ----------------------------------------------------------- #

    def show_model_evol(self, ax=None, zlin=None, flin=None,
                        TD=False, means=True,
                        elev=20, azim=260,
                        cmap='viridis_r', label='auto',
                        legend=False, leg_kwargs={},
                        fontsize='x-large', **kwargs):
        """ """
        if self.FIXED:
            raise TypeError("Model is fixed!")
        dgmap = plt.cm.get_cmap(cmap)
        Mlin = np.linspace(self.floor, self.ceil, 1000)
        if not TD:
            if ax is None:
                fig = plt.figure(figsize=[6, 4])
                ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])
            else:
                fig = ax.figure

            if (zlin is None) and (flin is not None):
                f_m, f_p = np.percentile(flin, [0, 100])
                for f in flin:
                    self.show_model_tot(ax=ax, model=False, show_hist=False,
                                        f=f, color=dgmap((f-f_m)/(f_p-f_m)),
                                        legend=False, means=means)
                ymax = np.max([self.plot_a(Mlin, f=f) for f in flin])
            elif (zlin is not None) and (flin is None):
                z_m, z_p = np.percentile(zlin, [0, 100])
                for z in zlin:
                    self.show_model_tot(ax=ax, model=False, show_hist=False,
                                        z=z, color=dgmap((z-z_m)/(z_p-z_m)),
                                        legend=False, means=means)
                ymax = np.max([self.plot_a(Mlin, z=z) for z in zlin])
            elif (zlin is None) and (flin is None):
                zlin = np.linspace(self.pd.redshifts.min(),
                                   self.pd.redshifts.max(), 50)
                z_m, z_p = np.percentile(zlin, [0, 100])
                for z in zlin:
                    self.show_model_tot(ax=ax, model=False, show_hist=False,
                                        z=z, color=dgmap((z-z_m)/(z_p-z_m)),
                                        legend=False, means=means)
                ymax = np.max([self.plot_a(Mlin, z=z) for z in zlin])
            elif (zlin is not None) and (flin is not None):
                raise NameError("`z` and `f` can't both be given")

            ax.set_xlabel('$M_*$', fontsize=fontsize)
            ax.set_ylabel('Probabilité', fontsize=fontsize)

            ax.set_ylim(0, 1.05*ymax)

            if legend:
                ax.legend(**leg_kwargs)

            return ax
        else:
            if ax is None:
                fig = plt.figure(figsize=[12, 8])
                ax = fig.add_axes([0.1, 0.12, 0.8, 0.8], projection='3d')
            else:
                fig = ax.figure

            if (zlin is None) and (flin is not None):
                ylin = flin
                X, Y = np.meshgrid(Mlin, ylin)
                zs = np.array(self.plot_a(np.ravel(X), f=np.ravel(Y)))
                ax.set_ylabel('$f$', fontsize='x-large')
            elif (zlin is not None) and (flin is None):
                ylin = zlin
                X, Y = np.meshgrid(Mlin, ylin)
                zs = np.array(self.plot_a(np.ravel(X), z=np.ravel(Y)))
                ax.set_ylabel('$z$', fontsize='x-large')
            elif (zlin is None) and (flin is None):
                ylin = np.linspace(self.pd.redshifts.min(),
                                   self.pd.redshifts.max(), 50)
                X, Y = np.meshgrid(Mlin, ylin)
                zs = np.array(self.plot_a(np.ravel(X), z=np.ravel(Y)))
                ax.set_ylabel('$z$', fontsize='x-large')
            elif (zlin is not None) and (flin is not None):
                raise NameError("`z` and `f` can't both be given")
            Z = zs.reshape(X.shape)

            ax.plot_surface(X, Y, Z,
                            cmap=dgmap,
                            facecolors=dgmap((Y-Y.min())/(Y.max()-Y.min()), 1),
                            lw=0, antialiased=True)

            ax.grid(False)
            ax.set_xlim(self.floor, self.ceil)
            # ax.set_ylim(df_nc.redshifts.min(), df_nc.redshifts.max())

            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            ax.view_init(elev=elev, azim=azim)

            ax.set_xlabel('$M_*$')
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel('Probabilité', fontsize='x-large', rotation=90)


# =========================================================================== #
#                            Dblegauss + dblegauss                            #
# =========================================================================== #


class Evol4G4M4S(Evol3G3M4S):
    '''4G4M4S'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['a', 'mu_1', 'sigma_1', 'mu_2', 'sigma_2']
    OLDPARAMETERS = ['b', 'mu_3', 'sigma_3', 'mu_4', 'sigma_4']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    GUESSVAL = [0.5, 8.5, 2, 10.0, 2,
                0.5, 9, 0.5, 10.5, 1]
    LIMVAL = [(0, 1), None, None, None, None,
              (0, 1), None, None, None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    def likelihood_y(self, x, dx,
                     a, mu_1, sigma_1, mu_2, sigma_2):
        '''Underlying mass distribution of young SNe Ia'''
        return a*self.gauss(x, dx, mu_1, sigma_1) + \
            (1-a)*self.gauss(x, dx, mu_2, sigma_2)

    def likelihood_o(self, x, dx,
                     b, mu_3, sigma_3, mu_4, sigma_4):
        '''Underlying mass distribution of old SNe Ia'''
        return b*self.gauss(x, dx, mu_3, sigma_3) + \
            (1-b)*self.gauss(x, dx, mu_4, sigma_4)

    def likelihood_tot(self, info, x, dx,
                       a, mu_1, sigma_1, mu_2, sigma_2,
                       b, mu_3, sigma_3, mu_4, sigma_4):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx,
                                      a, mu_1, sigma_1, mu_2, sigma_2) + \
            (1-info)*self.likelihood_o(x, dx,
                                       b, mu_3, sigma_3, mu_4, sigma_4)

    def loglikelihood(self,
                      a, mu_1, sigma_1, mu_2, sigma_2,
                      b, mu_3, sigma_3, mu_4, sigma_4):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.hostmass,
                                                    self.hostmass_err,
                                                    a, mu_1, sigma_1,
                                                    mu_2, sigma_2,
                                                    b, mu_3, sigma_3,
                                                    mu_4, sigma_4)))


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
    GUESSVAL = [9.5, 2, 0.5, 9, 0.5, 10.5, 1]
    LIMVAL = [None, None, (0, 1),
              None, None, None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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
#                                 Asym + Gaus                                 #
# =========================================================================== #


class Evol2G2M3S(Evol3G3M4S):
    '''Howell+asym'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu', 'sigmadown', 'sigmaup']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    GUESSVAL = [9.5, 1, 10.5, 1, 0.5]
    LIMVAL = [None, None,
              None, None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    def likelihood_o(self, x, dx, mu, sigmadown, sigmaup):
        '''Underlying mass distribution of old SNe Ia'''
        return self.asymgauss(x, dx, mu, sigmadown, sigmaup)

    def likelihood_tot(self, info, x, dx, mu_1, sigma_1,
                       mu, sigmadown, sigmaup):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_1, sigma_1) + \
            (1-info)*self.likelihood_o(x, dx, mu, sigmadown, sigmaup)

    def loglikelihood(self, mu_1, sigma_1, mu, sigmadown, sigmaup):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.hostmass,
                                                    self.hostmass_err,
                                                    mu_1, sigma_1,
                                                    mu, sigmadown, sigmaup)))

# =========================================================================== #
#                                 Asym + Asym                                 #
# =========================================================================== #


class Evol2G2M4S(Evol2G2M3S):
    '''Howell asym'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_y', 'sigmad_y', 'sigmau_y']
    OLDPARAMETERS = ['mu_o', 'sigmad_o', 'sigmau_o']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    GUESSVAL = [9.5, 1, 1, 10.5, 1, 0.5]
    LIMVAL = [None, None, None,
              None, None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = False

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    def likelihood_o(self, x, dx, mu_o, sigmad_o, sigmau_o):
        '''Underlying mass distribution of old SNe Ia'''
        return self.asymgauss(x, dx, mu_o, sigmad_o, sigmau_o)

    def likelihood_y(self, x, dx, mu_y, sigmad_y, sigmau_y):
        '''Underlying mass distribution of old SNe Ia'''
        return self.asymgauss(x, dx, mu_y, sigmad_y, sigmau_y)

    def likelihood_tot(self, info, x, dx,
                       mu_y, sigmad_y, sigmau_y,
                       mu_o, sigmad_o, sigmau_o):
        '''La fonction prenant en compte la probabilité d'être vieille/jeune'''
        return info*self.likelihood_y(x, dx, mu_y, sigmad_y, sigmau_y) + \
            (1-info)*self.likelihood_o(x, dx, mu_o, sigmad_o, sigmau_o)

    def loglikelihood(self, mu_y, sigmad_y, sigmau_y,
                      mu_o, sigmad_o, sigmau_o):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.info,
                                                    self.hostmass,
                                                    self.hostmass_err,
                                                    mu_y, sigmad_y, sigmau_y,
                                                    mu_o, sigmad_o, sigmau_o)))


# =========================================================================== #
#                                Gauss + gauss                                #
# =========================================================================== #


class Evol2G2M2S(Evol2G2M3S):
    '''Howell'''

    # =================================================================== #
    #                              Variables                              #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = ['mu_1', 'sigma_1']
    OLDPARAMETERS = ['mu_2', 'sigma_2']
    FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
                                         YOUNGPARAMETERS),
                               OLDPARAMETERS)
    GUESSVAL = [9.5, 1, 10.5, 1]
    LIMVAL = [None, None,
              None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
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

# =========================================================================== #
#                                 Beta + Beta                                 #
# =========================================================================== #


# class Evol2B(Evol2G2M2S):
#     '''2B'''
#
#     # =================================================================== #
#     #                              Parameters                             #
#     # =================================================================== #
#
#     GLOBALPARAMETERS = []
#     YOUNGPARAMETERS = [a_y, b_y, mu_y, sigma_y]
#     OLDPARAMETERS = [a_o, b_o, mu_o, sigma_o]
#     FREEPARAMETERS = np.append(np.append(GLOBALPARAMETERS,
#                                          YOUNGPARAMETERS),
#                                OLDPARAMETERS)
#     GUESSVAL = [10, 1]
#     LIMVAL = [None, None]
#     GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
#     FIXED = True

# =========================================================================== #
#                                    Gauss                                    #
# =========================================================================== #


class Evol1G1M1S(Evol2G2M2S):
    '''1G1M1S'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigma']
    GUESSVAL = [10, 1]
    LIMVAL = [None, None]
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
        return -2*np.sum(np.log(self.likelihood_tot(self.hostmass,
                                                    self.hostmass_err,
                                                    mu, sigma)))

# =========================================================================== #
#                                    Asymm                                    #
# =========================================================================== #


class Evol1G1M2S(Evol1G1M1S):
    '''1G1M2S'''

    # =================================================================== #
    #                              Parameters                             #
    # =================================================================== #

    GLOBALPARAMETERS = []
    YOUNGPARAMETERS = []
    OLDPARAMETERS = []
    FREEPARAMETERS = ['mu', 'sigmadown', 'sigmaup']
    GUESSVAL = [10, 1, 0.5]
    LIMVAL = [None, None, None]
    GUESS = {k: v for k, v in zip(FREEPARAMETERS, GUESSVAL)}
    FIXED = True

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               FITTER                                #
    # ------------------------------------------------------------------- #

    def likelihood_tot(self, x, dx, mu, sigmadown, sigmaup):
        '''La fonction de distribution'''
        return self.asymgauss(x, dx, mu, sigmadown, sigmaup)

    def loglikelihood(self, mu, sigmadown, sigmaup):
        '''La fonction à minimiser'''
        return -2*np.sum(np.log(self.likelihood_tot(self.hostmass,
                                                    self.hostmass_err,
                                                    mu, sigmadown, sigmaup)))
