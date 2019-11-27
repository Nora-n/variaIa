#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================== #
#                                                                             #
#                                  PREAMBULE                                  #
#                                                                             #
# =========================================================================== #

# =========================================================================== #
#                                    Pandas                                   #
# =========================================================================== #

d = pd.read_csv('../Data/data_cheat.csv', sep=' ', index_col='CID')
d_snf = pd.read_csv('../Data/lssfr_paper_full_sntable.csv', sep=',')

# =========================================================================== #
#                                 General Dict                                #
# =========================================================================== #

surveys = ['SNF', 'SDSS', 'PS1', 'SNLS', 'HST']

surv = {'SNF':  d_snf,
        'SDSS': d[d['IDSURVEY'] == 1],
        'PS1':  d[d['IDSURVEY'] == 15],
        'SNLS': d[d['IDSURVEY'] == 4],
        'HST':  d[d['IDSURVEY'].isin([101, 100, 106])]}

colors = {'SNF': 'orange',
          'SDSS': 'lime',
          'PS1': 'blue',
          'SNLS': 'red',
          'HST': 'purple'}

alpha3colors = {'SDSS': (0, 1, 0, .3),
                'SNLS': (1, 0, 0, .3),
                'PS1': (0, 0, 1, .3),
                'HST': (.5, 0, .5, .3),
                'SNF': (1, .647, 0, .3)}

alpha8colors = {'SDSS': (0, 1, 0, .8),
                'SNLS': (1, 0, 0, .8),
                'PS1': (0, 0, 1, .8),
                'HST': (.5, 0, .5, .8),
                'SNF': (1, .647, 0, .8)}

# =========================================================================== #
#                                                                             #
#                                 MASTERCLASS                                 #
#                                                                             #
# =========================================================================== #


class SimplePlots():
    """ """

    # =================================================================== #
    #                             Data choice                             #
    # =================================================================== #

    def choice_data(self, zc):
        with open(zc, 'rb') as f:
            if np.size(pickle.load(f)) == 5:
                self.z_lins, self.meds, self.stds,\
                    self.z_max, self.itsc = pickle.load(f)
            else:
                self.z_max = pickle.load(f)
        self.z_max['SNF'] = [10, 10, 10]
        self.z_max['HST'] = [10, 10, 10]

    # =================================================================== #
    #                               Data set                              #
    # =================================================================== #

    def set_data(self, su):
        self.survey = su
        # ----------------------------------------------------------- #
        #                           Raw data                          #
        # ----------------------------------------------------------- #

        if su == 'SNF':
            self.dataz = np.sort(surv[su]['host.zcmb'].values)
            self.datax = np.sort(surv[su]['salt2.X1'].values)
        else:
            self.dataz = np.sort(surv[su].zCMB.values)
            self.datax = np.sort(surv[su].x1.values)

        # ----------------------------------------------------------- #
        #                         To zmax Dict                        #
        # ----------------------------------------------------------- #

        zmax_cuts = dict()
        self.z_zcuts = dict()
        self.x1_zcuts = dict()

        for survey in surveys[1:]:
            zmax_cuts[survey] =\
                np.where(surv[survey].zCMB.values < self.z_max[survey][1])
            self.z_zcuts[survey] =\
                surv[survey].zCMB.values[zmax_cuts[survey]]
            self.x1_zcuts[survey] =\
                surv[survey].x1.values[zmax_cuts[survey]]

        zmax_cuts['SNF'] =\
            np.where(surv['SNF']['host.zcmb'].values < self.z_max['SNF'][1])
        self.z_zcuts['SNF'] =\
            surv['SNF']['host.zcmb'].values[zmax_cuts['SNF']]
        self.x1_zcuts['SNF'] =\
            surv['SNF']['salt2.X1'].values[zmax_cuts['SNF']]

        # ----------------------------------------------------------- #
        #                        From zmax Dict                       #
        # ----------------------------------------------------------- #

        zmin_cuts = dict()
        self.z_zmincuts = dict()
        self.x1_zmincuts = dict()

        for survey in surveys[1:]:
            zmin_cuts[survey] =\
                np.where(surv[survey].zCMB.values > self.z_max[survey][1])
            self.z_zmincuts[survey] =\
                surv[survey].zCMB.values[zmin_cuts[survey]]
            self.x1_zmincuts[survey] =\
                surv[survey].x1.values[zmin_cuts[survey]]

        zmin_cuts['SNF'] =\
            np.where(surv['SNF']['host.zcmb'].values > self.z_max['SNF'][1])
        self.z_zmincuts['SNF'] =\
            surv['SNF']['host.zcmb'].values[zmin_cuts['SNF']]
        self.x1_zmincuts['SNF'] =\
            surv['SNF']['salt2.X1'].values[zmin_cuts['SNF']]

        # ----------------------------------------------------------- #
        #                        In infsup Dict                       #
        # ----------------------------------------------------------- #

        zbtw_cuts = dict()
        self.z_zbtwcuts = dict()
        self.x1_zbtwcuts = dict()

        for survey in surveys[1:-1]:
            zbtw_cuts[survey] =\
                np.where((surv[survey].zCMB.values
                          > self.z_max[survey][0])
                         & (surv[survey].zCMB.values
                             < self.z_max[survey][2]))
            self.z_zbtwcuts[survey] =\
                surv[survey].zCMB.values[zbtw_cuts[survey]]
            self.x1_zbtwcuts[survey] =\
                surv[survey].x1.values[zbtw_cuts[survey]]

        # ----------------------------------------------------------- #
        #                       Out infsup Dict                       #
        # ----------------------------------------------------------- #

        zinf_cuts = dict()
        self.z_zinfcuts = dict()
        self.x1_zinfcuts = dict()

        for survey in surveys[1:-1]:
            zinf_cuts[survey] =\
                np.where(surv[survey].zCMB.values < self.z_max[survey][0])
            self.z_zinfcuts[survey] =\
                surv[survey].zCMB.values[zinf_cuts[survey]]
            self.x1_zinfcuts[survey] =\
                surv[survey].x1.values[zinf_cuts[survey]]

        zsup_cuts = dict()
        self.z_zsupcuts = dict()
        self.x1_zsupcuts = dict()

        for survey in surveys[1:-1]:
            zsup_cuts[survey] =\
                np.where(surv[survey].zCMB.values > self.z_max[survey][2])
            self.z_zsupcuts[survey] =\
                surv[survey].zCMB.values[zsup_cuts[survey]]
            self.x1_zsupcuts[survey] =\
                surv[survey].x1.values[zsup_cuts[survey]]

    # =================================================================== #
    #                               PLOTTING                              #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               Redshift                              #
    # ------------------------------------------------------------------- #

    def redshift_one(self, nb_z, show_zmax, show_infsup):
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        ax.hist(self.dataz, bins=nb_z,
                color=colors[self.survey], alpha=.5)

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=15,
                       top=True, right=True)

        ax.set_xlabel(r"$\mathrm{redshift}$ ", fontsize="x-large")
        ax.set_ylabel(r"$\mathrm{N}_\mathrm{SNe Ia}$ ", fontsize="x-large")

        if show_zmax:
            ax.vline(self.z_max[self.survey][1],
                     color=colors[self.survey], lw=2.0)

            if show_infsup:
                ax.axvspan(self.z_max[self.survey][0],
                           self.z_max[self.survey][2],
                           color=colors[self.survey],
                           alpha=.1, lw=2.0)

        plt.title(r'$\mathrm{Redshift\,\,distribution\,\,of\,\,}$'
                  + str(self.survey) +
                  r'$\mathrm{\,\,survey}$', fontsize='x-large')

    # =================================================================== #
    #                               Stretch                               #
    # =================================================================== #

    def stretch_one(self):
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        ax.scatter(self.z_zinfcuts[self.survey],
                   self.x1_zinfcuts[self.survey],
                   marker="o", s=50, lw=2,
                   edgecolor=alpha3colors[self.survey],
                   facecolor=alpha8colors[self.survey],
                   label=self.survey)

        ax.scatter(self.z_zbtwcuts[self.survey],
                   self.x1_zbtwcuts[self.survey],
                   marker="o", s=50, lw=2,
                   edgecolor=alpha3colors[self.survey],
                   facecolor=alpha3colors[self.survey],
                   label=self.survey)

        ax.scatter(self.z_zsupcuts[self.survey],
                   self.x1_zsupcuts[self.survey],
                   marker="o", s=50, lw=2,
                   edgecolor=alpha3colors[self.survey],
                   facecolor="None")

        ax.vline(self.z_max[self.survey][1],
                 color=colors[self.survey])
        ax.axvspan(self.z_max[self.survey][0],
                   self.z_max[self.survey][1],
                   color=colors[self.survey],
                   alpha=.1, lw=2.0)
        ax.axvspan(self.z_max[self.survey][1],
                   self.z_max[self.survey][2],
                   color=colors[self.survey],
                   alpha=.1, lw=2.0)
        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=15,
                       top=True, right=True)

        ax.set_xlabel(r'$\mathrm{redshift}$', fontsize='x-large')
        ax.set_ylabel(r'$x_1$', fontsize='x-large')

        ax.set_xlim(np.min(self.z_zinfcuts[self.survey])-1e-2,
                    np.max(self.z_zsupcuts[self.survey])+1e-2)
        ax.set_ylim(np.min(self.x1_zinfcuts[self.survey])-1e-1,
                    np.max(self.x1_zinfcuts[self.survey])+1e-1)

        plt.title(r'$\mathrm{Strech\,\,distribution\,\,of\,\,}$'
                  + str(self.survey)
                  + r'$\mathrm{\,\,survey\,\,for\,\,}$'
                  + str(self.z_max[self.survey][0])
                  + r'$< z_{\mathrm{max}} < $'
                  + str(self.z_max[self.survey][2]), fontsize='x-large')

    # =================================================================== #
    #                              Everything                             #
    # =================================================================== #

    def plot_all(self, nb_x, show_span, show_cdf, show_itsc, show_infsup):
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])
        ax2 = fig.add_axes([0.1, 0.92, 0.8, 0.2])

        ax2max = []

        for survey in surveys[1:-1]:
            if show_span:
                ax.scatter(self.z_zinfcuts[survey],
                           self.x1_zinfcuts[survey],
                           marker="o", s=50, lw=2,
                           edgecolor=alpha3colors[survey],
                           facecolor=alpha8colors[survey],
                           label=survey)

                ax.scatter(self.z_zbtwcuts[survey],
                           self.x1_zbtwcuts[survey],
                           marker="o", s=50, lw=2,
                           edgecolor=alpha3colors[survey],
                           facecolor=alpha3colors[survey])

                ax.scatter(self.z_zsupcuts[survey],
                           self.x1_zsupcuts[survey],
                           marker="o", s=50, lw=2,
                           edgecolor=alpha3colors[survey],
                           facecolor="None")

                ax.axvspan(self.z_max[survey][0],
                           self.z_max[survey][1],
                           color=colors[survey],
                           alpha=.1, lw=2.0)
                ax.axvspan(self.z_max[survey][1],
                           self.z_max[survey][2],
                           color=colors[survey],
                           alpha=.1, lw=2.0)

                ax2.axvspan(self.z_max[survey][0],
                            self.z_max[survey][1],
                            color=colors[survey],
                            alpha=.1, lw=2.0)
                ax2.axvspan(self.z_max[survey][1],
                            self.z_max[survey][2],
                            color=colors[survey],
                            alpha=.1, lw=2.0)
            else:
                ax.scatter(self.z_zcuts[survey],
                           self.x1_zcuts[survey],
                           marker="o", s=50, lw=2,
                           edgecolor=alpha3colors[survey],
                           facecolor=alpha8colors[survey],
                           label=survey)

                ax.scatter(self.z_zmincuts[survey],
                           self.x1_zmincuts[survey],
                           marker="o", s=50, lw=2,
                           edgecolor=alpha3colors[survey],
                           facecolor=None,
                           label=survey)

            ax.vline(self.z_max[survey][1],
                     color=colors[survey])
            ax2.vline(self.z_max[survey][1],
                      color=colors[survey], lw=1.0)

            ax2.hist(self.z_zcuts[survey], bins=nb_x,
                     color=colors[survey], alpha=.2)
            ax2.hist(self.z_zmincuts[survey], bins=nb_x,
                     color=colors[survey], alpha=.5, histtype='step')

            ax2max.append(np.max(np.histogram(self.z_zcuts[survey],
                                              bins=nb_x)[0]))
            ax2max.append(np.max(np.histogram(self.z_zmincuts[survey],
                                              bins=nb_x)[0]))

            if show_cdf:
                ax3 = ax2.twinx()

                ax3.plot(self.z_lins[survey],
                         self.meds[survey],
                         color=colors[survey])

                if show_itsc:
                    ax3.hline(self.itsc[survey][1],
                              color="0.3", lw=1.0)
                    ax3.plot(self.z_max[survey][1],
                             self.itsc[survey][1],
                             color="black", marker='o')

                    if show_infsup:
                        ax3.hline(self.itsc[survey][0],
                                  color="0.3", lw=1.0)
                        ax3.plot(self.z_max[survey][0],
                                 self.itsc[survey][0],
                                 color=".7", marker='o')

                        ax3.hline(self.itsc[survey][2],
                                  color="0.3", lw=1.0)
                        ax3.plot(self.z_max[survey][2],
                                 self.itsc[survey][2],
                                 color=".7", marker='o')

                ax3.set_ylim(0, 1.0)
                ax3.set_ylabel(r'$\mathrm{Poisson\,\,cdf}$', fontsize='large')

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=12,
                       top=True, right=True)

        ax.set_xlim(0, 1.1)
        ax.set_ylim(-3, 3)
        ax.set_xlabel(r"$\mathrm{redshift}$ ", fontsize="x-large")
        ax.set_ylabel(r"$\mathrm{x}_1$ ", fontsize="x-large")

        ax2.set_xlim(0, 1.1)
        ax2.set_ylim(0, np.max(ax2max))
        ax2.set_ylabel(r"$\mathrm{\# SNe Ia}$ ", fontsize="x-large")

        ax.legend(ncol=1, loc='lower left', fontsize='medium',
                  bbox_to_anchor=(0.82, 0.75))
