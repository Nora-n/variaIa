#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import optimize
from matplotlib.patches import Ellipse

c = 9.715611890751e-9  # in pc.s⁻¹

# =========================================================================== #
#                                                                             #
#                                 MASTERCLASS                                 #
#                                                                             #
# =========================================================================== #


class ellipse():
    """ """

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

    def set_param(self, param):
        """ Dict containing :
            alpha
            beta
            O_m
            O_L
            H0 <- MUST BE IN s⁻¹ !! Factor of 3.240779e-20 from km.Mpc⁻¹.s⁻¹
        """
        self.param = param

    def set_varia(self, M_abs, z_obs, m_max):
        """Self-explanatory"""
        self.M_abs = M_abs
        self.z_obs = z_obs
        self.m_max = m_max

    # ------------------------------------------------------------------- #
    #                               EXTFUNC                               #
    # ------------------------------------------------------------------- #

    def integrand(self, zp):
        """Integrand needed in d_L"""
        O_R = 1 - self.param['O_m'] - self.param['O_L']
        return((O_R*(1+zp)**4
                + self.param['O_m']*(1+zp)**3
                + self.param['O_L'])**(-1/2))

    def d_L(self, z):
        """The distance of a photon from emission to reception"""
        return((1+z)*c/self.param['H0']
               * integrate.quad(self.integrand, 0, z)[0])

    def m_obs(self, M_abs, x1, c, z):
        """Computing of observed magnitude"""
        return(5*np.log10(self.d_L(z)/10)
               + M_abs
               - self.param['alpha']*x1
               + self.param['beta']*c)

    # ------------------------------------------------------------------- #
    #                               SOLVER                                #
    # ------------------------------------------------------------------- #

    def m_cal(self):
        """Effective obsvd magnitude depending on M and z, for all x and c"""
        c_lin = np.linspace(-.4, .4, 100)
        x1_lin = 10*c_lin
        m_list = [[] for i in range(len(c_lin))]
        for i in range(len(c_lin)):
            m_list[i] = np.asarray([self.m_obs(self.M_abs,
                                               x1_lin[k],
                                               c_lin[i],
                                               self.z_obs)
                                    for k in range(len(x1_lin))])
        self.m_list = np.asarray(m_list)

        x1_max = []
        for i in range(len(x1_lin)):
            loc = np.where(self.m_list[i] < self.m_max)
            if np.size(loc) > 0:
                x1_max.append(x1_lin[np.min(loc)])
        self.x1_max = np.asarray(x1_max)
        self.c_max = c_lin[:len(x1_max)]

    def find_zmax(self, mag_lim, magabs_lim=-18.):
        return(self.find_z(mag_lim - magabs_lim))

    def find_z(self, distmod, cosmo=None):
        if cosmo is None:
            from astropy.cosmology import Planck15
            cosmo = Planck15
        return(optimize.fmin(lambda z:
                             np.abs(cosmo.distmod(z).value
                                    - distmod),
                             0.1, disp=0))

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

    def m_plot(self):
        fig = plt.figure(figsize=[16, 10])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        ax.add_patch(Ellipse((0, 0),
                     width=6,
                     height=.6,
                     ec="black",
                     fc="1"))

        ax.plot(self.x1_max,
                self.c_max,
                color="blue")

        ax.fill_between(self.x1_max,
                        self.c_max,
                        [-.4 for i in range(len(self.c_max))],
                        ec="1", fc="blue",
                        alpha=.05, zorder=2,
                        label=r'$m_{\mathrm{obs}} < m_{\mathrm{max}}$')

        if self.c_max[-1] == .4:
            xlin = np.linspace(self.x1_max[-1], 4, 2)
            ax.fill_between(xlin,
                            [.4 for i in range(len(xlin))],
                            [-.4 for i in range(len(xlin))],
                            ec="1", fc="blue",
                            alpha=.05, zorder=2)
            ax.vline(self.x1_max[-1],
                     color=(0, 0, 1, .025))

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize='x-large',
                       top=True, right=True)

        ax.set_xlabel(r'$x_1$', fontsize='xx-large')
        ax.set_ylabel(r'$c$', fontsize='xx-large')

        plt.legend(ncol=1, loc='upper left', fontsize='xx-large')

        plt.title(r'$\mathrm{Parameters\,\,where\,\,}m_{\mathrm{obs}} $'
                  + r'$< m_{\mathrm{max}}\mathrm{\,\,for\,\,}M_\mathrm{abs}$'
                  + r'$ = $' + str(self.M_abs)
                  + r'$\mathrm{,\,\,}m_{\mathrm{max}} = $' + str(self.m_max)
                  + r'$\mathrm{,\,\,and\,\,}z_{\mathrm{obs}} = $'
                  + str(self.z_obs),
                  fontsize='x-large')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-.4, .4)
